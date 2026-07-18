//! Per-edge, arbitrary-color border (letterbox/pillarbox) detection.
//!
//! `core/stages/crop.py`'s existing detector (FFmpeg `cropdetect`) is
//! luma-threshold-only: it treats "dark" as "border" and cannot see white,
//! gray, or otherwise colored padding. This crate replaces that per-frame
//! detection step (not `aggregate_crop_windows()`, which is unchanged and
//! still owns transition-exclusion/union aggregation -- see AGENTS.md's
//! Auto-crop section and `core/stages/crop.py`) with a detector that, per
//! sampled frame and per edge (top/bottom/left/right):
//!
//! 1. Computes the DOMINANT color of the outermost `strip_px`-deep strip of
//!    that edge and the fraction of strip pixels matching it within
//!    `tolerance` -- the edge's border color + "solidity" (how consistent/
//!    solid the border is, not just its color).
//! 2. If solidity < `solidity_min`, the edge has no solid border (border
//!    depth 0) -- this specifically keeps a blurred-video-background
//!    pillarboxed frame (a real, but non-solid, edge) from being cropped.
//! 3. Otherwise walks inward line by line while >= `majority` of the line's
//!    pixels are within `tolerance` of the dominant color -- a logo/overlay
//!    occupying a minority of a border line doesn't stop the walk. The
//!    first failing line is the content boundary.
//!
//! ## Dominant color: 4-bit-per-channel quantization
//!
//! Strip pixels are bucketed into `16^3` bins (4 bits/channel: `channel >>
//! 4`), and the largest bin's MEAN ACTUAL color (not the quantized bin
//! value) is reported as the dominant color. This is robust against slight
//! gradients/compression noise in an otherwise-solid border (which would
//! otherwise scatter exact-match pixel counts across many near-identical
//! colors) while still being cheap (one pass, integer bucketing, no
//! clustering).
//!
//! ## Color distance: max per-channel absolute difference
//!
//! Two colors "match" (for both solidity and the inward walk) iff
//! `max(|db|, |dg|, |dr|) <= tolerance`. Simple, fast (no sqrt/multiply),
//! and predictable to reason about/tune from `stages.crop.border_tolerance`.
//!
//! ## Walk depth cap
//!
//! The inward walk never goes deeper than 45% of the edge's dimension
//! (height for top/bottom, width for left/right) -- a "border" spanning
//! close to half the frame is not a border. Without this cap, a
//! pathological near-full-frame-solid-color frame (e.g. a black transition
//! frame) would walk almost all the way to the center and report a
//! near-full-depth border on every edge; `aggregate_crop_windows()`'s
//! existing transition-exclusion logic already handles solid-color
//! transition frames correctly (excludes them as a short-lived, isolated
//! run) as long as this detector doesn't itself produce a degenerate
//! all-black content rect for one -- the cap keeps that frame's *reported*
//! per-edge border modest instead of pathological, same spirit as
//! `cropdetect`'s own `round`/`limit` guards.
//!
//! ## Decoding
//!
//! Self-contained, like `avf_scenes`/`avf_hashing`: dimensions and frame
//! rate are probed internally via `ffprobe` (not passed in from the Python
//! side), then frames are decoded via a piped `ffmpeg -nostdin -i path
//! [-vf fps=sample_fps] -f rawvideo -pix_fmt bgr24 -` subprocess (stdin
//! detached -- see `avf_scenes`'s "tty corruption fix" note; this crate
//! never feeds ffmpeg interactive input). Unlike `avf_scenes`/`avf_hashing`
//! (which silently return empty/zero on any ffmpeg failure), this crate's
//! contract requires a real diagnosable error (ffmpeg death, non-zero exit,
//! or a truncated mid-frame read all raise `RuntimeError` in Python with an
//! ffmpeg stderr tail) -- stderr is therefore actually captured on a
//! background thread (bounded tail, same `StderrTail` shape as
//! `avf_framepipe`), not discarded to `Stdio::null()`.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::io::Read;
use std::process::{Child, ChildStderr, Command, Stdio};
use std::thread;
use std::thread::JoinHandle;

/// Bytes of ffmpeg stderr retained for error messages (tail, not full log) --
/// same bound and rationale as `avf_framepipe::StderrTail`.
const STDERR_TAIL_BYTES: usize = 4096;

/// Never walk a border more than this fraction of the edge's dimension --
/// see module docs' "Walk depth cap" section.
const MAX_WALK_FRACTION: f64 = 0.45;

// ---------------------------------------------------------------------
// stderr tail capture (background thread, joined once the child is reaped)
// ---------------------------------------------------------------------

struct StderrTail {
    handle: JoinHandle<Vec<u8>>,
}

impl StderrTail {
    fn spawn(mut stderr: ChildStderr) -> Self {
        let handle = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        });
        Self { handle }
    }

    /// Join the reader thread and return the trailing `STDERR_TAIL_BYTES` of
    /// captured stderr as a string.
    fn join_and_tail(self) -> String {
        let buf = self.handle.join().unwrap_or_default();
        let start = buf.len().saturating_sub(STDERR_TAIL_BYTES);
        String::from_utf8_lossy(&buf[start..]).trim().to_string()
    }
}

// ---------------------------------------------------------------------
// ffprobe: width/height/fps
// ---------------------------------------------------------------------

/// Probe `(width, height, fps)` via ffprobe. `None` if the probe fails or
/// dimensions can't be determined.
fn probe_dimensions_fps(ffprobe_path: &str, filepath: &str) -> Option<(u32, u32, f64)> {
    let output = Command::new(ffprobe_path)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1",
            "--",
            filepath,
        ])
        // ffprobe has no -nostdin flag; detaching stdin is the only lever
        // (matches avf_scenes/avf_hashing's own ffprobe calls).
        .stdin(Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut fps: f64 = 30.0;

    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "width" => width = value.trim().parse::<u32>().ok(),
            "height" => height = value.trim().parse::<u32>().ok(),
            "r_frame_rate" => {
                if let Some((num, den)) = value.split_once('/')
                    && let (Ok(n), Ok(d)) = (num.parse::<f64>(), den.parse::<f64>())
                    && d > 0.0
                {
                    fps = n / d;
                }
            }
            _ => {}
        }
    }

    if fps <= 0.0 {
        fps = 30.0;
    }

    match (width, height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => Some((w, h, fps)),
        _ => None,
    }
}

// ---------------------------------------------------------------------
// Per-frame border detection
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Edge {
    Top,
    Bottom,
    Left,
    Right,
}

const EDGES: [Edge; 4] = [Edge::Top, Edge::Bottom, Edge::Left, Edge::Right];

impl Edge {
    fn name(self) -> &'static str {
        match self {
            Edge::Top => "top",
            Edge::Bottom => "bottom",
            Edge::Left => "left",
            Edge::Right => "right",
        }
    }

    /// Dimension the walk is bounded by (height for top/bottom, width for
    /// left/right).
    fn dimension(self, w: u32, h: u32) -> u32 {
        match self {
            Edge::Top | Edge::Bottom => h,
            Edge::Left | Edge::Right => w,
        }
    }
}

#[inline]
fn pixel_at(buf: &[u8], w: u32, x: u32, y: u32) -> (u8, u8, u8) {
    let idx = ((y * w + x) * 3) as usize;
    (buf[idx], buf[idx + 1], buf[idx + 2])
}

#[inline]
fn color_matches(a: (u8, u8, u8), b: (u8, u8, u8), tolerance: u8) -> bool {
    let d0 = (a.0 as i32 - b.0 as i32).unsigned_abs();
    let d1 = (a.1 as i32 - b.1 as i32).unsigned_abs();
    let d2 = (a.2 as i32 - b.2 as i32).unsigned_abs();
    let tol = tolerance as u32;
    d0 <= tol && d1 <= tol && d2 <= tol
}

/// Collect the outermost `depth`-deep strip of pixels for `edge` (depth
/// clamped to the edge's own dimension so a strip request deeper than the
/// frame is harmless).
fn strip_pixels(buf: &[u8], w: u32, h: u32, edge: Edge, depth: u32) -> Vec<(u8, u8, u8)> {
    let depth = depth.min(edge.dimension(w, h)).max(1);
    let mut pixels = Vec::new();
    match edge {
        Edge::Top => {
            for y in 0..depth {
                for x in 0..w {
                    pixels.push(pixel_at(buf, w, x, y));
                }
            }
        }
        Edge::Bottom => {
            for y in (h - depth)..h {
                for x in 0..w {
                    pixels.push(pixel_at(buf, w, x, y));
                }
            }
        }
        Edge::Left => {
            for x in 0..depth {
                for y in 0..h {
                    pixels.push(pixel_at(buf, w, x, y));
                }
            }
        }
        Edge::Right => {
            for x in (w - depth)..w {
                for y in 0..h {
                    pixels.push(pixel_at(buf, w, x, y));
                }
            }
        }
    }
    pixels
}

/// Dominant color (largest 4-bit-per-channel quantization bin's mean actual
/// color) + solidity (fraction of `pixels` within `tolerance` of it).
fn dominant_color_and_solidity(pixels: &[(u8, u8, u8)], tolerance: u8) -> ((u8, u8, u8), f64) {
    if pixels.is_empty() {
        return ((0, 0, 0), 0.0);
    }

    // Bin key: 4 bits/channel packed into a u16 (b<<8 | g<<4 | r).
    let mut bins: HashMap<u16, (u64, u64, u64, u64)> = HashMap::new();
    for &(b, g, r) in pixels {
        let key = ((b >> 4) as u16) << 8 | ((g >> 4) as u16) << 4 | (r >> 4) as u16;
        let entry = bins.entry(key).or_insert((0, 0, 0, 0));
        entry.0 += b as u64;
        entry.1 += g as u64;
        entry.2 += r as u64;
        entry.3 += 1;
    }

    let (_, &(sum_b, sum_g, sum_r, count)) = bins
        .iter()
        .max_by_key(|&(_, &(_, _, _, count))| count)
        .expect("bins is non-empty since pixels is non-empty");

    let dominant = (
        (sum_b / count) as u8,
        (sum_g / count) as u8,
        (sum_r / count) as u8,
    );

    let matching = pixels
        .iter()
        .filter(|&&p| color_matches(p, dominant, tolerance))
        .count();
    let solidity = matching as f64 / pixels.len() as f64;

    (dominant, solidity)
}

/// Fraction of a single line's pixels (at `depth` from `edge`) matching
/// `color` within `tolerance`.
fn line_match_fraction(
    buf: &[u8],
    w: u32,
    h: u32,
    edge: Edge,
    depth: u32,
    color: (u8, u8, u8),
    tolerance: u8,
) -> f64 {
    let (matching, total) = match edge {
        Edge::Top | Edge::Bottom => {
            let y = if edge == Edge::Top {
                depth
            } else {
                h - 1 - depth
            };
            let matching = (0..w)
                .filter(|&x| color_matches(pixel_at(buf, w, x, y), color, tolerance))
                .count();
            (matching, w as usize)
        }
        Edge::Left | Edge::Right => {
            let x = if edge == Edge::Left {
                depth
            } else {
                w - 1 - depth
            };
            let matching = (0..h)
                .filter(|&y| color_matches(pixel_at(buf, w, x, y), color, tolerance))
                .count();
            (matching, h as usize)
        }
    };
    if total == 0 {
        0.0
    } else {
        matching as f64 / total as f64
    }
}

/// Walk inward from `edge` while >= `majority` of each line's pixels match
/// `color`, bounded by `MAX_WALK_FRACTION` of the edge's dimension. Returns
/// the number of consecutive matching lines from the edge (the "border
/// depth").
fn walk_border(
    buf: &[u8],
    w: u32,
    h: u32,
    edge: Edge,
    color: (u8, u8, u8),
    tolerance: u8,
    majority: f64,
) -> u32 {
    let dim = edge.dimension(w, h);
    let cap = ((dim as f64) * MAX_WALK_FRACTION).floor() as u32;
    let mut border = 0u32;
    for depth in 0..cap {
        let frac = line_match_fraction(buf, w, h, edge, depth, color, tolerance);
        if frac >= majority {
            border += 1;
        } else {
            break;
        }
    }
    border
}

struct EdgeResult {
    color: (u8, u8, u8),
    solidity: f64,
    border_px: u32,
}

struct FrameResult {
    t: f64,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    edges: HashMap<&'static str, EdgeResult>,
}

/// Analyze one decoded BGR24 frame: per-edge dominant color/solidity/walk,
/// then derive the content rect from the four border depths.
#[allow(clippy::too_many_arguments)]
fn analyze_frame(
    buf: &[u8],
    w: u32,
    h: u32,
    t: f64,
    strip_px: u32,
    tolerance: u8,
    majority: f64,
    solidity_min: f64,
) -> FrameResult {
    let mut edges: HashMap<&'static str, EdgeResult> = HashMap::new();

    for &edge in &EDGES {
        let strip = strip_pixels(buf, w, h, edge, strip_px);
        let (color, solidity) = dominant_color_and_solidity(&strip, tolerance);
        let border_px = if solidity >= solidity_min {
            walk_border(buf, w, h, edge, color, tolerance, majority)
        } else {
            0
        };
        edges.insert(
            edge.name(),
            EdgeResult {
                color,
                solidity,
                border_px,
            },
        );
    }

    let left = edges["left"].border_px;
    let right = edges["right"].border_px;
    let top = edges["top"].border_px;
    let bottom = edges["bottom"].border_px;

    let content_w = w.saturating_sub(left + right).max(1);
    let content_h = h.saturating_sub(top + bottom).max(1);

    FrameResult {
        t,
        x: left,
        y: top,
        w: content_w,
        h: content_h,
        edges,
    }
}

// ---------------------------------------------------------------------
// ffmpeg decode + detect loop
// ---------------------------------------------------------------------

#[derive(Debug)]
enum DetectError {
    Spawn(String),
    /// ffmpeg exited non-zero, or the stream ended mid-frame -- both are a
    /// diagnosable failure with a stderr tail attached.
    Failed(String),
}

fn spawn_ffmpeg(
    ffmpeg_path: &str,
    filepath: &str,
    sample_fps: f64,
) -> Result<(Child, std::process::ChildStdout, StderrTail), String> {
    let mut args: Vec<String> = vec![
        "-v".into(),
        "error".into(),
        // Non-interactive: this ffmpeg is never fed keyboard commands.
        // Without it, ffmpeg raw-modes its controlling tty (if any) to poll
        // stdin for interactive keys and won't restore it if killed/crashed
        // -- stdin(Stdio::null()) below is belt-and-suspenders for the same
        // reason. See avf_scenes's identical note ("tty corruption fix").
        "-nostdin".into(),
        "-i".into(),
        filepath.into(),
    ];
    if sample_fps > 0.0 {
        args.push("-vf".into());
        args.push(format!("fps={sample_fps}"));
    }
    args.extend([
        "-f".into(),
        "rawvideo".into(),
        "-pix_fmt".into(),
        "bgr24".into(),
        "-".into(),
    ]);

    let mut child = Command::new(ffmpeg_path)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("avf_borders: failed to spawn ffmpeg ({ffmpeg_path}): {e}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "avf_borders: failed to capture ffmpeg stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "avf_borders: failed to capture ffmpeg stderr".to_string())?;
    let stderr_tail = StderrTail::spawn(stderr);

    Ok((child, stdout, stderr_tail))
}

#[allow(clippy::too_many_arguments)]
fn run_detection(
    filepath: &str,
    ffmpeg_path: &str,
    ffprobe_path: &str,
    sample_fps: f64,
    strip_px: u32,
    tolerance: u8,
    majority: f64,
    solidity_min: f64,
) -> Result<Vec<FrameResult>, DetectError> {
    let Some((width, height, probed_fps)) = probe_dimensions_fps(ffprobe_path, filepath) else {
        return Err(DetectError::Spawn(format!(
            "avf_borders: could not probe dimensions for '{filepath}' via ffprobe ({ffprobe_path})"
        )));
    };
    let effective_fps = if sample_fps > 0.0 {
        sample_fps
    } else {
        probed_fps
    };

    let (mut child, mut stdout, stderr_tail) =
        spawn_ffmpeg(ffmpeg_path, filepath, sample_fps).map_err(DetectError::Spawn)?;

    let frame_size = (width as usize) * (height as usize) * 3;
    let mut readbuf = vec![0u8; frame_size];
    let mut results = Vec::new();
    let mut frame_idx: u64 = 0;
    let mut truncated = false;

    loop {
        let mut total = 0usize;
        let mut hit_eof = false;
        loop {
            match stdout.read(&mut readbuf[total..frame_size]) {
                Ok(0) => {
                    hit_eof = true;
                    break;
                }
                Ok(n) => {
                    total += n;
                    if total >= frame_size {
                        break;
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    // A genuine I/O error (not a clean EOF) is always a
                    // failure worth surfacing, even if it happened to land
                    // exactly on a frame boundary (total == 0).
                    hit_eof = true;
                    truncated = true;
                    break;
                }
            }
        }

        if hit_eof {
            if total != 0 {
                truncated = true;
            }
            break;
        }

        let t = frame_idx as f64 / effective_fps;
        results.push(analyze_frame(
            &readbuf,
            width,
            height,
            t,
            strip_px,
            tolerance,
            majority,
            solidity_min,
        ));
        frame_idx += 1;
    }

    // Always reap the child -- no zombies, regardless of outcome.
    let status = child.wait();
    let tail = stderr_tail.join_and_tail();

    if truncated {
        return Err(DetectError::Failed(format!(
            "avf_borders: input ended mid-frame while decoding (truncated rawvideo stream); ffmpeg stderr tail:\n{tail}"
        )));
    }

    match status {
        Ok(s) if s.success() => Ok(results),
        Ok(s) => Err(DetectError::Failed(format!(
            "avf_borders: ffmpeg exited with {s}; stderr tail:\n{tail}"
        ))),
        Err(e) => Err(DetectError::Failed(format!(
            "avf_borders: ffmpeg process could not be reaped: {e}; stderr tail:\n{tail}"
        ))),
    }
}

// ---------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------

fn frame_result_to_pydict<'py>(
    py: Python<'py>,
    frame: &FrameResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("t", frame.t)?;
    dict.set_item("x", frame.x)?;
    dict.set_item("y", frame.y)?;
    dict.set_item("w", frame.w)?;
    dict.set_item("h", frame.h)?;

    let edges_dict = PyDict::new(py);
    for &edge in &EDGES {
        let e = &frame.edges[edge.name()];
        let edge_dict = PyDict::new(py);
        edge_dict.set_item("color", (e.color.0, e.color.1, e.color.2))?;
        edge_dict.set_item("solidity", e.solidity)?;
        edge_dict.set_item("border_px", e.border_px)?;
        edges_dict.set_item(edge.name(), edge_dict)?;
    }
    dict.set_item("edges", edges_dict)?;

    Ok(dict)
}

/// Python-callable entry point: per-sampled-frame, per-edge dominant-color
/// border detection. See module docs for the algorithm.
///
/// `ffmpeg_path`/`ffprobe_path` are resolved by the caller (honoring
/// `ffmpeg.binary` config, `PATH`, etc. -- see `core/ffmpeg_utils.py`)
/// rather than re-resolved here, matching `avf_scenes`/`avf_hashing`'s
/// convention. `ffprobe_path` isn't in the spec's headline signature but is
/// required to keep dimension-probing self-contained in Rust (see module
/// docs' "Decoding" section) -- every other Rust-backed call site in this
/// project (`_detect_scene_changes_rust`, `compute_video_phash`) already
/// resolves and passes both paths, so this follows that precedent.
#[pyfunction]
#[pyo3(signature = (
    path,
    ffmpeg_path,
    sample_fps=0.0,
    strip_px=4,
    tolerance=24,
    majority=0.80,
    solidity_min=0.60,
    ffprobe_path="ffprobe".to_string(),
))]
#[allow(clippy::too_many_arguments)]
fn detect_border_frames(
    py: Python<'_>,
    path: String,
    ffmpeg_path: String,
    sample_fps: f64,
    strip_px: u32,
    tolerance: u8,
    majority: f64,
    solidity_min: f64,
    ffprobe_path: String,
) -> PyResult<Py<PyList>> {
    let result = py.detach(move || {
        run_detection(
            &path,
            &ffmpeg_path,
            &ffprobe_path,
            sample_fps,
            strip_px.max(1),
            tolerance,
            majority,
            solidity_min,
        )
    });

    let frames = match result {
        Ok(frames) => frames,
        Err(DetectError::Spawn(msg)) => return Err(PyRuntimeError::new_err(msg)),
        Err(DetectError::Failed(msg)) => return Err(PyRuntimeError::new_err(msg)),
    };

    let list = PyList::empty(py);
    for frame in &frames {
        list.append(frame_result_to_pydict(py, frame)?)?;
    }
    Ok(list.unbind())
}

#[pymodule]
fn avf_borders(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_border_frames, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command as StdCommand;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("avf_borders_test_{pid}_{nanos}_{name}"));
        p
    }

    fn ffmpeg_available() -> bool {
        StdCommand::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Black 320x240 canvas with 320x180 testsrc2 content centered
    /// (letterboxed top/bottom by 30px each).
    fn make_letterbox_clip(pad_color: &str, frames: u32) -> PathBuf {
        let path = unique_temp_path("letterbox.mp4");
        let filter = format!(
            "color=c={pad_color}:s=320x240:d=1[bg];\
             testsrc2=size=320x180:rate=5[fg];\
             [bg][fg]overlay=x=0:y=30:shortest=1"
        );
        let status = StdCommand::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &filter,
                "-frames:v",
                &frames.to_string(),
                "-pix_fmt",
                "yuv420p",
                path.to_str().unwrap(),
            ])
            .status()
            .expect("failed to run ffmpeg to build letterbox fixture");
        assert!(
            status.success(),
            "ffmpeg letterbox fixture generation failed"
        );
        path
    }

    /// White letterbox with a small colored drawbox logo in the bottom
    /// band, `logo_width_frac` of the frame's width wide.
    fn make_letterbox_with_logo(logo_width_frac: f64) -> PathBuf {
        let path = unique_temp_path("letterbox_logo.mp4");
        let logo_w = (320.0 * logo_width_frac).round() as u32;
        let filter = format!(
            "color=c=white:s=320x240:d=1[bg];\
             testsrc2=size=320x180:rate=5[fg];\
             [bg][fg]overlay=x=0:y=30:shortest=1[base];\
             [base]drawbox=x=10:y=200:w={logo_w}:h=20:color=red@1.0:t=fill"
        );
        let status = StdCommand::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &filter,
                "-frames:v",
                "3",
                "-pix_fmt",
                "yuv420p",
                path.to_str().unwrap(),
            ])
            .status()
            .expect("failed to run ffmpeg to build logo fixture");
        assert!(status.success(), "ffmpeg logo fixture generation failed");
        path
    }

    /// Full-frame random noise -- no edge of this clip has ANY dominant
    /// color above `solidity_min` (unlike `testsrc2`, which has flat-ish
    /// color bars that can spuriously satisfy a low `solidity_min`/small
    /// `strip_px` right at its outer edge).
    fn make_full_frame_clip() -> PathBuf {
        let path = unique_temp_path("full.mp4");
        let status = StdCommand::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "nullsrc=size=320x240:rate=5,geq=random(1)*255:random(2)*255:random(3)*255",
                "-frames:v",
                "3",
                "-pix_fmt",
                "yuv420p",
                path.to_str().unwrap(),
            ])
            .status()
            .expect("failed to run ffmpeg to build full-frame fixture");
        assert!(
            status.success(),
            "ffmpeg full-frame fixture generation failed"
        );
        path
    }

    #[test]
    fn black_letterbox_detected() {
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        let path = make_letterbox_clip("black", 2);
        Python::attach(|py| {
            let frames = run_detection(
                path.to_str().unwrap(),
                "ffmpeg",
                "ffprobe",
                0.0,
                4,
                24,
                0.90,
                0.60,
            )
            .expect("detection failed");
            assert!(!frames.is_empty());
            let f = &frames[0];
            let top = &f.edges["top"];
            let bottom = &f.edges["bottom"];
            assert!(
                (25..=35).contains(&top.border_px),
                "top border_px={} expected ~30",
                top.border_px
            );
            assert!(
                (25..=35).contains(&bottom.border_px),
                "bottom border_px={} expected ~30",
                bottom.border_px
            );
            assert!(top.solidity > 0.95, "top solidity={}", top.solidity);
            assert!(
                top.color.0 < 10 && top.color.1 < 10 && top.color.2 < 10,
                "top color={:?} expected near-black",
                top.color
            );
            let _ = py;
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn white_letterbox_detected_with_correct_color() {
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        let path = make_letterbox_clip("white", 2);
        Python::attach(|py| {
            let frames = run_detection(
                path.to_str().unwrap(),
                "ffmpeg",
                "ffprobe",
                0.0,
                4,
                24,
                0.90,
                0.60,
            )
            .expect("detection failed");
            let f = &frames[0];
            let top = &f.edges["top"];
            assert!(
                (25..=35).contains(&top.border_px),
                "top border_px={} expected ~30 (cropdetect cannot see this -- white padding)",
                top.border_px
            );
            assert!(
                top.color.0 > 245 && top.color.1 > 245 && top.color.2 > 245,
                "top color={:?} expected near-white",
                top.color
            );
            let _ = py;
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn small_logo_does_not_stop_majority_walk() {
        // ~8% width drawbox: 92% of the line still matches -> holds at the
        // default majority=0.90.
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        let path = make_letterbox_with_logo(0.08);
        Python::attach(|py| {
            let frames = run_detection(
                path.to_str().unwrap(),
                "ffmpeg",
                "ffprobe",
                0.0,
                4,
                24,
                0.90,
                0.60,
            )
            .expect("detection failed");
            let f = &frames[0];
            let bottom = &f.edges["bottom"];
            assert!(
                (25..=35).contains(&bottom.border_px),
                "bottom border_px={} expected ~30 despite the small logo (majority=0.90 tolerates an 8%-wide overlay)",
                bottom.border_px
            );
            let _ = py;
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn large_logo_stops_default_majority_but_not_relaxed_majority() {
        // A 25%-width drawbox fails 25% of the line -> only 75% matches,
        // below the default majority=0.90, so the walk stops at the logo's
        // row. Passing majority=0.70 explicitly, though, tolerates it and
        // the walk reaches the true border depth -- pins the majority
        // semantics both ways.
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        let path = make_letterbox_with_logo(0.25);
        Python::attach(|py| {
            let strict = run_detection(
                path.to_str().unwrap(),
                "ffmpeg",
                "ffprobe",
                0.0,
                4,
                24,
                0.90,
                0.60,
            )
            .expect("detection failed");
            let relaxed = run_detection(
                path.to_str().unwrap(),
                "ffmpeg",
                "ffprobe",
                0.0,
                4,
                24,
                0.70,
                0.60,
            )
            .expect("detection failed");

            let strict_bottom = strict[0].edges["bottom"].border_px;
            let relaxed_bottom = relaxed[0].edges["bottom"].border_px;

            // The logo sits at y=200 in a 240-tall frame -- 40px from the
            // bottom edge, i.e. depth 39 counting from the bottom. The
            // strict (default) walk must stop at or before that row; the
            // relaxed walk must walk past it, all the way to ~30.
            assert!(
                strict_bottom < 39,
                "strict (majority=0.90) bottom border_px={strict_bottom} expected to stop at the 25%-wide logo row"
            );
            assert!(
                relaxed_bottom >= 25,
                "relaxed (majority=0.70) bottom border_px={relaxed_bottom} expected to tolerate the logo and reach ~30"
            );
            assert!(
                relaxed_bottom > strict_bottom,
                "relaxed walk ({relaxed_bottom}) should walk deeper than strict ({strict_bottom})"
            );
            let _ = py;
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn no_border_on_busy_full_frame_content() {
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        let path = make_full_frame_clip();
        Python::attach(|py| {
            let frames = run_detection(
                path.to_str().unwrap(),
                "ffmpeg",
                "ffprobe",
                0.0,
                4,
                24,
                0.90,
                0.60,
            )
            .expect("detection failed");
            let f = &frames[0];
            for edge in ["top", "bottom", "left", "right"] {
                let e = &f.edges[edge];
                assert_eq!(
                    e.border_px, 0,
                    "{edge} border_px={} expected 0 on busy full-frame content (solidity={})",
                    e.border_px, e.solidity
                );
            }
            let _ = py;
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn corrupt_input_errors_without_hang_or_zombie() {
        let result = run_detection(
            "/nonexistent/path/definitely_not_a_video.mp4",
            "ffmpeg",
            "ffprobe",
            0.0,
            4,
            24,
            0.90,
            0.60,
        );
        assert!(matches!(result, Err(DetectError::Spawn(_))));
    }

    #[test]
    fn dominant_color_quantization_picks_largest_bin_mean() {
        // Two near-black pixels (slightly different, same 4-bit bin) and
        // one far pixel in a different bin -- dominant should be the
        // near-black bin's mean, not the far outlier.
        let pixels = vec![(2u8, 3u8, 1u8), (4u8, 2u8, 3u8), (250u8, 250u8, 250u8)];
        let (color, solidity) = dominant_color_and_solidity(&pixels, 24);
        assert!(color.0 < 10 && color.1 < 10 && color.2 < 10, "{color:?}");
        assert!((solidity - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn color_matches_respects_tolerance() {
        assert!(color_matches((10, 10, 10), (20, 20, 20), 10));
        assert!(!color_matches((10, 10, 10), (21, 10, 10), 10));
    }
}
