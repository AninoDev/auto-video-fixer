//! Rust rewrite of video perceptual hashing / duplicate detection
//! (`compute_video_hash`/`compute_video_dhash`/`hash_similarity` in
//! `core/analysis.py`) per `docs/REQUIREMENTS.md` R5.2.
//!
//! ## Algorithm choice: pHash (DCT-based), not a literal ahash/dhash port
//!
//! The original Python implementation computed both an average hash (ahash)
//! and a difference hash (dhash) per sampled frame, then combined per-hash
//! bits via majority vote across frames. R5.2 explicitly lifts the
//! backward-compatibility constraint (no stored hash values exist anywhere,
//! confirmed with the user 2026-07-12), so this rewrite picks a single,
//! stronger algorithm rather than porting either of those bit-for-bit:
//! **pHash**, the classic DCT-based perceptual hash (Krawetz's
//! phash.org algorithm). Reasoning:
//!
//! - ahash (mean threshold) and dhash (adjacent-pixel gradient) are both
//!   *spatial-domain* hashes -- sensitive to exactly the kind of variation
//!   real-world near-duplicate video files have: re-encode ringing/blocking
//!   artifacts, minor resize/crop shifts, and small color-grading changes all
//!   perturb individual pixel values directly.
//! - pHash instead hashes the low-frequency 2D DCT coefficients of the
//!   downscaled frame. Low frequencies encode coarse luminance structure
//!   (the actual picture content) and are naturally robust to the
//!   high-frequency noise that compression artifacts and resampling
//!   introduce -- exactly the property a duplicate-detector needs. This is
//!   the standard "when accuracy matters more than raw speed" choice, and
//!   speed is not a real constraint once this is compiled Rust code either
//!   way (see the differential test's timings).
//!
//! `img_hash` (crates.io) was evaluated as a pre-built implementation but
//! rejected: it operates on `image::GenericImageView`, so using it would
//! pull in the `image` crate and its per-format codec dependencies (~24
//! transitive crates as of this writing) purely to wrap raw grayscale bytes
//! we already have from our own ffmpeg pipe (see `avf_scenes` for the
//! established pattern) -- image *decoding* is not something this crate
//! needs, since ffmpeg already handles it upstream. pHash itself is a small,
//! well-understood algorithm (downscale, 2D DCT, threshold against the
//! median/mean of the low-frequency block), so hand-rolling it here avoids
//! that dependency bloat for no loss of correctness. The Python-side
//! fallback (`core/analysis.py`, used when this extension isn't built)
//! implements the identical algorithm in pure NumPy for the same reason --
//! see that module's docstring.
//!
//! ## Frame sampling and combination
//!
//! Frames are sampled the same way the original `compute_video_hash` did:
//! `num_frames` (default 30) evenly spaced frames across the video, selected
//! via ffprobe's frame-count estimate and an ffmpeg `select` filter (rather
//! than per-frame seeks, which are far slower for many small seeks). Each
//! sampled frame is decoded pre-scaled to 32x32 grayscale directly by
//! ffmpeg, DCT'd, and reduced to a 64-bit hash. Per-frame hashes are combined
//! into one video-level hash via majority-vote bit combination (same
//! aggregation strategy as the original code) -- this is still a reasonable
//! choice here: it's cheap, and it makes the video hash robust to any single
//! sampled frame being a transient outlier (e.g. a mid-scene flash) without
//! needing a more elaborate aggregation scheme.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::io::Read;
use std::process::{Command, Stdio};

/// Side length of the frame ffmpeg decodes each sampled frame down to before
/// the DCT. 32x32 is the standard pHash working resolution.
const SIDE: usize = 32;
const FRAME_SIZE: usize = SIDE * SIDE;
/// Side length of the retained low-frequency DCT block (top-left corner).
const LOW_FREQ: usize = 8;

/// Probe the total (estimated) frame count via ffprobe. Returns 0 if it
/// can't be determined -- callers fall back to a fixed sampling stride in
/// that case, matching `avf_scenes`'s "unknown frame count" degradation.
fn probe_total_frames(ffprobe_path: &str, filepath: &str) -> u64 {
    let Ok(output) = Command::new(ffprobe_path)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1",
            "--",
            filepath,
        ])
        .output()
    else {
        return 0;
    };

    if !output.status.success() {
        return 0;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut fps: f64 = 30.0;
    let mut nb_frames: Option<u64> = None;
    let mut duration: Option<f64> = None;

    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "r_frame_rate" => {
                if let Some((num, den)) = value.split_once('/')
                    && let (Ok(n), Ok(d)) = (num.parse::<f64>(), den.parse::<f64>())
                    && d > 0.0
                {
                    fps = n / d;
                }
            }
            "nb_frames" => nb_frames = value.trim().parse::<u64>().ok(),
            "duration" => duration = value.trim().parse::<f64>().ok(),
            _ => {}
        }
    }

    nb_frames.unwrap_or_else(|| match duration {
        Some(d) if d > 0.0 => (d * fps).round() as u64,
        _ => 0,
    })
}

/// Decode up to `num_frames` evenly-spaced frames, each pre-scaled by ffmpeg
/// to `SIDE`x`SIDE` grayscale, and return their raw pixel buffers.
fn sample_frames(
    ffmpeg_path: &str,
    ffprobe_path: &str,
    filepath: &str,
    num_frames: u64,
) -> Vec<[u8; FRAME_SIZE]> {
    let num_frames = num_frames.max(1);
    let total_frames = probe_total_frames(ffprobe_path, filepath);
    let step = if total_frames > 0 {
        (total_frames / num_frames).max(1)
    } else {
        1
    };

    let vf = if step > 1 {
        format!("select='not(mod(n\\,{step}))',scale={SIDE}:{SIDE}:flags=bilinear,format=gray")
    } else {
        format!("scale={SIDE}:{SIDE}:flags=bilinear,format=gray")
    };

    let mut child = match Command::new(ffmpeg_path)
        .args([
            "-v", "error", "-i", filepath, "-vf", &vf, "-vsync", "vfr", "-f", "rawvideo",
            "-pix_fmt", "gray", "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut stdout = match child.stdout.take() {
        Some(s) => s,
        None => return Vec::new(),
    };

    let mut frames = Vec::new();
    let mut buf = [0u8; FRAME_SIZE];
    while (frames.len() as u64) < num_frames {
        if stdout.read_exact(&mut buf).is_err() {
            break;
        }
        frames.push(buf);
    }

    // Drain and drop the rest so ffmpeg isn't left blocked writing to a
    // closed-but-not-yet-reaped pipe (only matters when total_frames was
    // unknown and step==1, so ffmpeg emits more frames than we need).
    let mut sink = [0u8; 65536];
    while stdout.read(&mut sink).unwrap_or(0) > 0 {}
    let _ = child.wait();

    frames
}

/// Build the NxN orthonormal DCT-II basis matrix (row-major, `matrix[u][x]`).
fn dct_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    let alpha0 = (1.0 / n as f64).sqrt();
    let alpha = (2.0 / n as f64).sqrt();
    for u in 0..n {
        let a = if u == 0 { alpha0 } else { alpha };
        for x in 0..n {
            m[u * n + x] =
                a * (std::f64::consts::PI / n as f64 * (x as f64 + 0.5) * u as f64).cos();
        }
    }
    m
}

/// `n x n` matrix multiply, both operands and result row-major.
fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = a[i * n + j];
        }
    }
    out
}

/// Compute the 64-bit pHash of one `SIDE`x`SIDE` grayscale frame.
/// Bit `i` (MSB-first, row-major over the 8x8 low-frequency block)
/// corresponds to whether that DCT coefficient exceeds the mean of the
/// low-frequency block *excluding* the DC term (matches the reference
/// phash.org algorithm: the DC coefficient dominates the magnitude and
/// would otherwise skew the threshold).
fn phash_frame(frame: &[u8; FRAME_SIZE], dct: &[f64], dct_t: &[f64]) -> u64 {
    let mut img = vec![0.0f64; FRAME_SIZE];
    for (i, &p) in frame.iter().enumerate() {
        img[i] = p as f64;
    }

    let step1 = matmul(dct, &img, SIDE);
    let coeffs = matmul(&step1, dct_t, SIDE);

    let mut low = [0.0f64; LOW_FREQ * LOW_FREQ];
    for row in 0..LOW_FREQ {
        for col in 0..LOW_FREQ {
            low[row * LOW_FREQ + col] = coeffs[row * SIDE + col];
        }
    }

    let sum_excl_dc: f64 = low.iter().skip(1).sum();
    let mean = sum_excl_dc / (low.len() - 1) as f64;

    let mut hash: u64 = 0;
    for (i, &v) in low.iter().enumerate() {
        if v > mean {
            hash |= 1u64 << (63 - i);
        }
    }
    hash
}

/// Combine multiple 64-bit per-frame hashes into one video-level hash via
/// majority-vote bit combination (ties -> 0, matching the original Python
/// `bits.count("1") > len(bits) / 2` semantics).
fn combine_majority(hashes: &[u64]) -> u64 {
    if hashes.is_empty() {
        return 0;
    }
    let n = hashes.len();
    let mut combined: u64 = 0;
    for bit in 0..64 {
        let mask = 1u64 << bit;
        let ones = hashes.iter().filter(|h| *h & mask != 0).count();
        if ones * 2 > n {
            combined |= mask;
        }
    }
    combined
}

/// Python-callable entry point: sample `num_frames` evenly-spaced frames
/// from `filepath`, pHash each, and combine via majority vote into one
/// 64-bit video hash, returned as a 16-character lowercase hex string (empty
/// string on any decode failure, matching the original functions' `""`
/// sentinel for "couldn't hash this file").
#[pyfunction]
#[pyo3(signature = (filepath, num_frames=30, ffmpeg_path="ffmpeg".to_string(), ffprobe_path="ffprobe".to_string()))]
fn compute_phash_rs(
    py: Python<'_>,
    filepath: String,
    num_frames: u64,
    ffmpeg_path: String,
    ffprobe_path: String,
) -> PyResult<String> {
    let hash = py.detach(move || {
        let frames = sample_frames(&ffmpeg_path, &ffprobe_path, &filepath, num_frames);
        if frames.is_empty() {
            return None;
        }
        let dct = dct_matrix(SIDE);
        let dct_t = transpose(&dct, SIDE);
        let per_frame: Vec<u64> = frames
            .iter()
            .map(|f| phash_frame(f, &dct, &dct_t))
            .collect();
        Some(combine_majority(&per_frame))
    });

    Ok(match hash {
        Some(h) => format!("{h:016x}"),
        None => String::new(),
    })
}

/// Python-callable entry point: similarity (0.0-1.0, higher = more similar)
/// between two hex-encoded 64-bit pHash strings, based on Hamming distance.
/// Returns 0.0 for empty/malformed/mismatched-length input, matching the
/// original `hash_similarity`'s degenerate-input handling.
#[pyfunction]
fn hash_similarity_rs(hash1: &str, hash2: &str) -> PyResult<f64> {
    if hash1.is_empty() || hash2.is_empty() || hash1.len() != hash2.len() {
        return Ok(0.0);
    }
    let (Ok(a), Ok(b)) = (
        u64::from_str_radix(hash1, 16),
        u64::from_str_radix(hash2, 16),
    ) else {
        return Err(PyValueError::new_err("hash must be a hex string"));
    };
    let distance = (a ^ b).count_ones();
    Ok(1.0 - (distance as f64 / 64.0))
}

#[pymodule]
fn avf_hashing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_phash_rs, m)?)?;
    m.add_function(wrap_pyfunction!(hash_similarity_rs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_matrix_is_orthonormal() {
        // D * D^T should be (very close to) the identity matrix.
        let d = dct_matrix(SIDE);
        let dt = transpose(&d, SIDE);
        let product = matmul(&d, &dt, SIDE);
        for i in 0..SIDE {
            for j in 0..SIDE {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i * SIDE + j] - expected).abs() < 1e-9,
                    "D*D^T[{i}][{j}] = {}, expected {expected}",
                    product[i * SIDE + j]
                );
            }
        }
    }

    #[test]
    fn identical_frames_hash_identically() {
        let dct = dct_matrix(SIDE);
        let dct_t = transpose(&dct, SIDE);
        let mut frame = [0u8; FRAME_SIZE];
        for (i, p) in frame.iter_mut().enumerate() {
            *p = ((i * 37) % 256) as u8;
        }
        let h1 = phash_frame(&frame, &dct, &dct_t);
        let h2 = phash_frame(&frame, &dct, &dct_t);
        assert_eq!(h1, h2);
    }

    #[test]
    fn flat_frame_and_noisy_frame_differ() {
        let dct = dct_matrix(SIDE);
        let dct_t = transpose(&dct, SIDE);
        let flat = [128u8; FRAME_SIZE];
        let mut checkerboard = [0u8; FRAME_SIZE];
        for row in 0..SIDE {
            for col in 0..SIDE {
                checkerboard[row * SIDE + col] = if (row + col) % 2 == 0 { 0 } else { 255 };
            }
        }
        let h_flat = phash_frame(&flat, &dct, &dct_t);
        let h_checker = phash_frame(&checkerboard, &dct, &dct_t);
        let distance = (h_flat ^ h_checker).count_ones();
        assert!(
            distance > 20,
            "expected very different hashes, got distance={distance}"
        );
    }

    #[test]
    fn majority_vote_ties_resolve_to_zero() {
        // Two hashes disagreeing on bit 0: majority-of-2 tie -> 0.
        let combined = combine_majority(&[1u64 << 63, 0]);
        assert_eq!(combined, 0);
    }

    #[test]
    fn hash_similarity_identical_is_one() {
        assert_eq!(
            hash_similarity_rs("00ff00ff00ff00ff", "00ff00ff00ff00ff").unwrap(),
            1.0
        );
    }

    #[test]
    fn hash_similarity_opposite_is_zero() {
        assert_eq!(
            hash_similarity_rs("0000000000000000", "ffffffffffffffff").unwrap(),
            0.0
        );
    }
}
