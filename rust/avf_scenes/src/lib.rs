//! Rust rewrite of `_detect_scene_changes()` (`core/analysis.py`) per
//! `docs/REQUIREMENTS.md` R5.1.
//!
//! Decodes a video via a piped `ffmpeg` subprocess (grayscale, downscaled to
//! 320x180 rawvideo frames on ffmpeg's own stdout), computes the mean
//! per-pixel absolute luma difference between consecutive frames (mirroring
//! `cv2.absdiff(...).mean() / 255.0`), and applies the same cut-detection /
//! min-duration-absorption / near-miss-tracking logic as the Python
//! implementation. See that function's docstring for the full metric
//! semantics and calibration numbers -- this module must match it, not
//! reinterpret it.
//!
//! Frame timestamps are derived as `frame_idx / fps` (a piped rawvideo
//! stream carries no per-frame PTS) -- this matches the Python
//! implementation's own fallback branch (used whenever `cv2.CAP_PROP_POS_MSEC`
//! isn't available) and is exact for constant-frame-rate sources, which is
//! what the calibration fixture and all currently-supported inputs are.

use pyo3::Py;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::io::Read;
use std::process::{Command, Stdio};

const FRAME_W: usize = 320;
const FRAME_H: usize = 180;
const FRAME_SIZE: usize = FRAME_W * FRAME_H;

/// One detected scene: (start_time, end_time, confidence).
type SceneTuple = (f64, f64, f64);
/// One near-miss candidate: (time, diff_score).
type NearMiss = (f64, f64);

struct DetectResult {
    scenes: Vec<SceneTuple>,
    cut_scores: Vec<f64>,
    near_misses: Vec<NearMiss>,
}

/// Probe fps (and, if available, an estimated total frame count) via ffprobe.
/// Returns `(fps, total_frames_estimate)`. `total_frames_estimate` is 0 if it
/// can't be determined (matching the Python implementation's "unknown frame
/// count" fallback, which switches to a fixed report interval).
fn probe_fps_and_frames(ffprobe_path: &str, filepath: &str) -> Option<(f64, u64)> {
    let output = Command::new(ffprobe_path)
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
        .ok()?;

    if !output.status.success() {
        return None;
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
            "nb_frames" => {
                nb_frames = value.trim().parse::<u64>().ok();
            }
            "duration" => {
                duration = value.trim().parse::<f64>().ok();
            }
            _ => {}
        }
    }

    if fps <= 0.0 {
        fps = 30.0;
    }

    let total_frames = nb_frames.unwrap_or_else(|| match duration {
        Some(d) if d > 0.0 => (d * fps).round() as u64,
        _ => 0,
    });

    Some((fps, total_frames))
}

/// Core decode/diff/detect loop. Runs entirely off the GIL except for the
/// brief moments it calls back into `progress_callback`.
fn run_detection(
    filepath: &str,
    threshold: f64,
    min_duration_sec: f64,
    ffmpeg_path: &str,
    ffprobe_path: &str,
    progress_callback: Option<&Py<PyAny>>,
) -> DetectResult {
    let empty = DetectResult {
        scenes: Vec::new(),
        cut_scores: Vec::new(),
        near_misses: Vec::new(),
    };

    let Some((fps, total_frames)) = probe_fps_and_frames(ffprobe_path, filepath) else {
        return empty;
    };

    let report_interval: u64 = if total_frames > 0 {
        (total_frames / 20).max(1)
    } else {
        200
    };

    let mut child = match Command::new(ffmpeg_path)
        .args([
            "-v",
            "error",
            "-i",
            filepath,
            "-vf",
            "format=gray,scale=320:180:flags=bilinear",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return empty,
    };

    let mut stdout = match child.stdout.take() {
        Some(s) => s,
        None => return empty,
    };

    let mut scenes: Vec<SceneTuple> = Vec::new();
    let mut cut_scores: Vec<f64> = Vec::new();
    let mut near_misses: Vec<NearMiss> = Vec::new();
    let near_miss_floor = threshold / 4.0;

    let mut prev: Option<Vec<u8>> = None;
    let mut buf = vec![0u8; FRAME_SIZE];
    let mut frame_idx: u64 = 0;
    let mut scene_start: f64 = 0.0;
    let mut current_time: f64 = 0.0;

    loop {
        if stdout.read_exact(&mut buf).is_err() {
            break;
        }

        current_time = frame_idx as f64 / fps;

        if let Some(ref p) = prev {
            let mut sum: u64 = 0;
            for i in 0..FRAME_SIZE {
                let a = p[i] as i32;
                let b = buf[i] as i32;
                sum += (a - b).unsigned_abs() as u64;
            }
            let diff_score = (sum as f64 / FRAME_SIZE as f64) / 255.0;

            if diff_score > threshold {
                if current_time - scene_start >= min_duration_sec {
                    scenes.push((scene_start, current_time, diff_score.min(1.0)));
                    cut_scores.push(diff_score);
                    scene_start = current_time;
                }
            } else if diff_score > near_miss_floor {
                near_misses.push((current_time, diff_score));
            }
        }

        prev = Some(buf.clone());
        frame_idx += 1;

        if let Some(cb) = progress_callback
            && frame_idx.is_multiple_of(report_interval)
        {
            let detail = if total_frames > 0 {
                let pct = ((frame_idx as f64 / total_frames as f64) * 100.0).min(100.0) as i64;
                format!("{}% (frame {}/{})", pct, frame_idx, total_frames)
            } else {
                format!("frame {}", frame_idx)
            };
            Python::attach(|py| {
                let _ = cb.call1(py, ("scene_detection", detail));
            });
        }
    }

    let _ = child.wait();

    if scene_start < current_time {
        let final_duration = current_time - scene_start;
        if final_duration >= min_duration_sec || scenes.is_empty() {
            scenes.push((scene_start, current_time, 0.5));
        }
    }

    DetectResult {
        scenes,
        cut_scores,
        near_misses,
    }
}

/// Python-callable entry point.
///
/// Returns `(scenes, cut_scores, near_misses)`:
///   - `scenes`: `list[(start_time, end_time, confidence)]`, matching
///     `SceneEvent` construction order/semantics in `core/analysis.py`.
///   - `cut_scores`: raw `diff_score` of each detected cut (NOT including the
///     trailing scene's placeholder 0.5 confidence) -- used for the
///     min/median/max log summary.
///   - `near_misses`: `list[(time, diff_score)]` for every frame pair that
///     scored in `(threshold/4, threshold]` -- feed to Python's
///     `_select_near_misses()` unchanged.
///
/// `ffmpeg_path`/`ffprobe_path` are resolved by the caller (honoring
/// `ffmpeg.binary` config, `PATH`, etc. -- see `core/ffmpeg_utils.py`) rather
/// than re-resolved here, so this stays consistent with the rest of the
/// project's FFmpeg-path handling.
#[pyfunction]
#[pyo3(signature = (filepath, threshold, min_duration_sec, ffmpeg_path, ffprobe_path, progress_callback=None))]
#[allow(clippy::too_many_arguments)]
fn detect_scene_changes_rs(
    py: Python<'_>,
    filepath: String,
    threshold: f64,
    min_duration_sec: f64,
    ffmpeg_path: String,
    ffprobe_path: String,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<(Vec<SceneTuple>, Vec<f64>, Vec<NearMiss>)> {
    let result = py.detach(move || {
        run_detection(
            &filepath,
            threshold,
            min_duration_sec,
            &ffmpeg_path,
            &ffprobe_path,
            progress_callback.as_ref(),
        )
    });
    Ok((result.scenes, result.cut_scores, result.near_misses))
}

#[pymodule]
fn avf_scenes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_scene_changes_rs, m)?)?;
    Ok(())
}
