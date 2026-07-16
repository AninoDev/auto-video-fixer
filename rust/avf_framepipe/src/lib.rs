//! Threaded, bounded-channel ffmpeg frame I/O for Python AI stages.
//!
//! `FrameReader` and `FrameWriter` each own a background OS thread plus a
//! piped `ffmpeg` subprocess, and hand frames across a bounded
//! (`std::sync::mpsc::sync_channel`) queue so a Python-side inference loop
//! can pull/push frames without ever blocking on the ffmpeg pipe directly.
//! Inference itself stays pure Python/PyTorch -- this crate only replaces
//! the "shell out to ffmpeg and shuttle rawvideo bytes" plumbing that used
//! to run inline on the calling thread (see `ai/frame_processor.py`'s
//! `FrameProcessor`/`StreamingVideoWriter`/`_AsyncFrameWriter`, which this
//! is intended to eventually back -- that wiring is a separate follow-up
//! task, not part of this crate).
//!
//! ## Threading model
//!
//! - `FrameReader::new()` spawns `ffmpeg -i <path> -f rawvideo -pix_fmt
//!   bgr24 -` and a decode thread that `read_exact`s frames into a *reused*
//!   per-thread buffer, copies each decoded frame into its own `Vec<u8>`,
//!   batches `chunk_size` of them, and pushes the batch through a
//!   `sync_channel(read_ahead)`. `next_batch()` pulls one batch and converts
//!   each frame into an independent, Python-owned `numpy.ndarray` (a single
//!   memcpy: the per-frame `Vec<u8>` is moved into the array's backing
//!   storage via `IntoPyArray`, not copied again).
//! - `FrameWriter::new()` spawns an ffmpeg encoder reading rawvideo from
//!   stdin and a writer thread that drains a `sync_channel(write_queue)` of
//!   frame batches and streams them into ffmpeg's stdin.
//! - All blocking channel/process operations happen with the GIL released
//!   (`Python::detach`); the GIL is only reacquired to build/consume numpy
//!   arrays.
//!
//! ## Buffer strategy (v1)
//!
//! One memcpy per frame into a Python-owned array. No lease/release
//! protocol, no pooling beyond the reused decode-thread read buffer -- see
//! the approved plan for the rationale (simplicity first; a future version
//! can add zero-copy buffer leasing if profiling shows the extra memcpy
//! matters relative to GPU inference time).
//!
//! ## Future extension point (not implemented here)
//!
//! Decode options are currently just `path`/`width`/`height`. A future v2
//! could introduce a `DecodeOpts`-style struct with e.g. `hwaccel:
//! Option<String>` to opt into NVDEC/VAAPI hardware decode -- deferred
//! because it needs its own error-handling story (hwaccel init failures
//! should probably fall back to software decode rather than hard-erroring)
//! and isn't needed for the v1 threaded-pipe correctness this crate
//! establishes.

use numpy::{IntoPyArray, PyArray3, PyArrayMethods, PyReadonlyArray3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::io::{Read, Write};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Bytes of ffmpeg stderr retained for error messages (tail, not full log).
const STDERR_TAIL_BYTES: usize = 2048;
/// Poll interval while waiting for a child process to be reaped without
/// holding the shared `Mutex<Child>` for the whole wait.
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(5);
/// Upper bound on how long a natural (non-killed) reap is allowed to poll
/// for before this crate force-kills the process itself as a self-heal
/// (should essentially never trigger: ffmpeg exits within milliseconds of
/// closing its piped stdout/stdin).
const WAIT_SELF_HEAL_TIMEOUT: Duration = Duration::from_secs(5);
/// Upper bound on how long `close()`/`Drop` will drain a channel waiting for
/// the producer/consumer thread to observe disconnect and exit.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------
// Shared helpers: stderr tail capture, bounded child wait/kill.
// ---------------------------------------------------------------------

/// Continuously drains a child's stderr pipe on a background thread (so a
/// chatty ffmpeg process can never deadlock on a full stderr pipe buffer)
/// and retains only the last `STDERR_TAIL_BYTES` for error reporting.
struct StderrTail {
    buf: Arc<Mutex<Vec<u8>>>,
    handle: Option<JoinHandle<()>>,
}

impl StderrTail {
    fn spawn(stderr: ChildStderr) -> Self {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let buf_thread = buf.clone();
        let handle = thread::spawn(move || {
            let mut reader = stderr;
            let mut chunk = [0u8; 4096];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => {
                        let mut b = buf_thread.lock().unwrap();
                        b.extend_from_slice(&chunk[..n]);
                        let len = b.len();
                        if len > STDERR_TAIL_BYTES {
                            let excess = len - STDERR_TAIL_BYTES;
                            b.drain(0..excess);
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            buf,
            handle: Some(handle),
        }
    }

    fn tail_string(&self) -> String {
        let b = self.buf.lock().unwrap();
        String::from_utf8_lossy(&b).trim().to_string()
    }

    /// Join the background reader thread. Safe to call multiple times.
    fn join(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for StderrTail {
    fn drop(&mut self) {
        self.join();
    }
}

/// Block (in short polling increments, so a concurrent `force_kill_and_reap`
/// from another thread can always acquire the lock) until the child exits,
/// returning its exit status. Self-heals by force-killing after
/// `WAIT_SELF_HEAL_TIMEOUT` in the pathological case where the process never
/// exits on its own.
fn wait_with_lock(child: &Arc<Mutex<Child>>) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + WAIT_SELF_HEAL_TIMEOUT;
    loop {
        {
            let mut guard = child.lock().unwrap();
            if let Ok(Some(status)) = guard.try_wait() {
                return Some(status);
            }
        }
        if Instant::now() >= deadline {
            let mut guard = child.lock().unwrap();
            let _ = guard.kill();
            return guard.wait().ok();
        }
        thread::sleep(WAIT_POLL_INTERVAL);
    }
}

/// Kill (if still running) and reap the child. Idempotent and safe to call
/// even if the process already exited on its own. Used by `close()` and the
/// `Drop` backstop on both `FrameReader` and `FrameWriter`.
fn force_kill_and_reap(child: &Arc<Mutex<Child>>) {
    let mut guard = child.lock().unwrap();
    let _ = guard.kill();
    let _ = guard.wait();
}

/// Drain `rx` (discarding messages) until the sending side disconnects,
/// bounded by `DRAIN_TIMEOUT`. Used to unblock a producer/consumer thread
/// that might be mid-`send`/`recv` when `close()`/`Drop` wants to shut down
/// early, without the drainer itself hanging forever.
fn drain_until_disconnected<T>(rx: &Receiver<T>) {
    let deadline = Instant::now() + DRAIN_TIMEOUT;
    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(_) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {
                if Instant::now() >= deadline {
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// FrameReader
// ---------------------------------------------------------------------

enum ReaderMsg {
    Batch(Vec<Vec<u8>>),
    Eos,
    Err(String),
}

enum ReaderTerminal {
    Eos,
    Err(String),
}

/// Outcome of attempting to read exactly one frame's worth of bytes from a
/// stream, distinguishing a clean frame-boundary EOF from a truncated
/// (mid-frame) EOF -- error contract case 4/5.
enum FrameReadOutcome {
    Full(Vec<u8>),
    CleanEof,
    Truncated,
    Io(std::io::Error),
}

fn read_one_frame(stdout: &mut impl Read, frame_size: usize, buf: &mut [u8]) -> FrameReadOutcome {
    let mut total = 0usize;
    while total < frame_size {
        match stdout.read(&mut buf[total..frame_size]) {
            Ok(0) => {
                return if total == 0 {
                    FrameReadOutcome::CleanEof
                } else {
                    FrameReadOutcome::Truncated
                };
            }
            Ok(n) => total += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return FrameReadOutcome::Io(e),
        }
    }
    FrameReadOutcome::Full(buf[..frame_size].to_vec())
}

#[allow(clippy::too_many_arguments)]
fn decode_thread_body(
    stdout: std::process::ChildStdout,
    stderr_tail: StderrTail,
    child: Arc<Mutex<Child>>,
    frame_size: usize,
    chunk_size: usize,
    tx: SyncSender<ReaderMsg>,
    frames_read: Arc<AtomicU64>,
) {
    let mut stdout = stdout;
    let mut readbuf = vec![0u8; frame_size];
    let mut batch: Vec<Vec<u8>> = Vec::with_capacity(chunk_size);
    let mut truncated = false;
    let mut io_err: Option<std::io::Error> = None;

    loop {
        match read_one_frame(&mut stdout, frame_size, &mut readbuf) {
            FrameReadOutcome::Full(data) => {
                batch.push(data);
                if batch.len() >= chunk_size {
                    let sent = std::mem::replace(&mut batch, Vec::with_capacity(chunk_size));
                    let n = sent.len() as u64;
                    if tx.send(ReaderMsg::Batch(sent)).is_err() {
                        // Receiver (Python side) is gone -- stop decoding.
                        drop(stderr_tail);
                        return;
                    }
                    frames_read.fetch_add(n, Ordering::SeqCst);
                }
            }
            FrameReadOutcome::CleanEof => break,
            FrameReadOutcome::Truncated => {
                truncated = true;
                break;
            }
            FrameReadOutcome::Io(e) => {
                io_err = Some(e);
                break;
            }
        }
    }

    if !batch.is_empty() {
        let n = batch.len() as u64;
        if tx.send(ReaderMsg::Batch(batch)).is_ok() {
            frames_read.fetch_add(n, Ordering::SeqCst);
        }
    }

    let status = wait_with_lock(&child);

    let final_msg = if truncated {
        ReaderMsg::Err(format!(
            "avf_framepipe: input ended mid-frame while decoding (truncated rawvideo stream); ffmpeg stderr tail:\n{}",
            stderr_tail.tail_string()
        ))
    } else if let Some(e) = io_err {
        ReaderMsg::Err(format!(
            "avf_framepipe: I/O error reading decoded frames: {e}; ffmpeg stderr tail:\n{}",
            stderr_tail.tail_string()
        ))
    } else {
        match status {
            Some(s) if s.success() => ReaderMsg::Eos,
            Some(s) => ReaderMsg::Err(format!(
                "avf_framepipe: ffmpeg exited with {s}; stderr tail:\n{}",
                stderr_tail.tail_string()
            )),
            None => ReaderMsg::Err(format!(
                "avf_framepipe: ffmpeg process could not be reaped; stderr tail:\n{}",
                stderr_tail.tail_string()
            )),
        }
    };

    let _ = tx.send(final_msg);
}

fn frame_to_array(
    py: Python<'_>,
    buf: Vec<u8>,
    height: usize,
    width: usize,
) -> PyResult<Py<PyArray3<u8>>> {
    let arr1 = buf.into_pyarray(py);
    let arr3 = arr1.reshape([height, width, 3]).map_err(|e| {
        PyRuntimeError::new_err(format!("avf_framepipe: frame reshape failed: {e}"))
    })?;
    Ok(arr3.unbind())
}

/// Threaded ffmpeg frame reader. See module docs for the threading model.
#[pyclass]
struct FrameReader {
    width: usize,
    height: usize,
    // Wrapped in a Mutex solely to satisfy pyo3's Send+Sync requirement for
    // pyclass fields (`mpsc::Receiver` is Send but not Sync); access is
    // always single-threaded from Python's perspective (the GIL serializes
    // calls into this object), so the lock never contends.
    rx: Mutex<Receiver<ReaderMsg>>,
    handle: Option<JoinHandle<()>>,
    child: Arc<Mutex<Child>>,
    frames_read: Arc<AtomicU64>,
    terminal: Option<ReaderTerminal>,
    closed: bool,
}

#[pymethods]
impl FrameReader {
    #[new]
    #[pyo3(signature = (path, width, height, chunk_size=25, read_ahead=2, ffmpeg_path="ffmpeg".to_string()))]
    fn new(
        path: String,
        width: usize,
        height: usize,
        chunk_size: usize,
        read_ahead: usize,
        ffmpeg_path: String,
    ) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyRuntimeError::new_err(
                "avf_framepipe: width and height must be > 0",
            ));
        }
        let chunk_size = chunk_size.max(1);
        let read_ahead = read_ahead.max(1);
        let frame_size = width * height * 3;

        let mut child = Command::new(&ffmpeg_path)
            .args([
                "-v", "error", "-nostdin", "-i", &path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "avf_framepipe: failed to spawn ffmpeg ({ffmpeg_path}): {e}"
                ))
            })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            PyRuntimeError::new_err("avf_framepipe: failed to capture ffmpeg stdout")
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            PyRuntimeError::new_err("avf_framepipe: failed to capture ffmpeg stderr")
        })?;
        let stderr_tail = StderrTail::spawn(stderr);

        let child = Arc::new(Mutex::new(child));
        let (tx, rx) = sync_channel::<ReaderMsg>(read_ahead);
        let frames_read = Arc::new(AtomicU64::new(0));

        let thread_child = child.clone();
        let thread_frames_read = frames_read.clone();
        let handle = thread::spawn(move || {
            decode_thread_body(
                stdout,
                stderr_tail,
                thread_child,
                frame_size,
                chunk_size,
                tx,
                thread_frames_read,
            );
        });

        Ok(FrameReader {
            width,
            height,
            rx: Mutex::new(rx),
            handle: Some(handle),
            child,
            frames_read,
            terminal: None,
            closed: false,
        })
    }

    /// Pull the next batch of decoded frames. Returns `None` on clean EOS.
    /// Raises `RuntimeError` (with an ffmpeg stderr tail) on decode failure.
    fn next_batch(&mut self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyArray3<u8>>>>> {
        if self.closed {
            return Ok(None);
        }
        if let Some(t) = &self.terminal {
            return match t {
                ReaderTerminal::Eos => Ok(None),
                ReaderTerminal::Err(m) => Err(PyRuntimeError::new_err(m.clone())),
            };
        }

        let msg = py.detach(|| self.rx.lock().unwrap().recv());
        match msg {
            Ok(ReaderMsg::Batch(frames)) => {
                let mut out = Vec::with_capacity(frames.len());
                for f in frames {
                    out.push(frame_to_array(py, f, self.height, self.width)?);
                }
                Ok(Some(out))
            }
            Ok(ReaderMsg::Eos) => {
                self.terminal = Some(ReaderTerminal::Eos);
                Ok(None)
            }
            Ok(ReaderMsg::Err(e)) => {
                self.terminal = Some(ReaderTerminal::Err(e.clone()));
                Err(PyRuntimeError::new_err(e))
            }
            Err(_) => {
                let msg = "avf_framepipe: decode thread ended unexpectedly".to_string();
                self.terminal = Some(ReaderTerminal::Err(msg.clone()));
                Err(PyRuntimeError::new_err(msg))
            }
        }
    }

    fn frames_read(&self) -> u64 {
        self.frames_read.load(Ordering::SeqCst)
    }

    /// Idempotent: kills + reaps ffmpeg and joins the decode thread.
    fn close(&mut self, py: Python<'_>) {
        if self.closed {
            return;
        }
        self.closed = true;
        let child = self.child.clone();
        let handle = self.handle.take();
        py.detach(|| {
            force_kill_and_reap(&child);
            drain_until_disconnected(&self.rx.lock().unwrap());
            if let Some(h) = handle {
                let _ = h.join();
            }
        });
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        force_kill_and_reap(&self.child);
        drain_until_disconnected(&self.rx.lock().unwrap());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ---------------------------------------------------------------------
// FrameWriter
// ---------------------------------------------------------------------

enum WriteMsg {
    Batch(Vec<Vec<u8>>),
}

struct WriterState {
    error: Option<String>,
    success: Option<bool>,
}

#[allow(clippy::too_many_arguments)]
fn writer_thread_body(
    stdin: std::process::ChildStdin,
    stderr_tail: StderrTail,
    child: Arc<Mutex<Child>>,
    rx: Receiver<WriteMsg>,
    frames_written: Arc<AtomicU64>,
    state: Arc<Mutex<WriterState>>,
) {
    let mut stdin = stdin;
    let mut poisoned = false;

    while let Ok(WriteMsg::Batch(frames)) = rx.recv() {
        if poisoned {
            continue;
        }
        for f in frames {
            if let Err(e) = stdin.write_all(&f) {
                let mut st = state.lock().unwrap();
                st.error = Some(format!(
                    "avf_framepipe: ffmpeg stdin write failed: {e}; stderr tail:\n{}",
                    stderr_tail.tail_string()
                ));
                poisoned = true;
                break;
            }
            frames_written.fetch_add(1, Ordering::SeqCst);
        }
    }

    drop(stdin);
    let status = wait_with_lock(&child);

    let mut st = state.lock().unwrap();
    if st.error.is_none() {
        match status {
            Some(s) if s.success() => st.success = Some(true),
            Some(s) => {
                st.error = Some(format!(
                    "avf_framepipe: ffmpeg exited with {s}; stderr tail:\n{}",
                    stderr_tail.tail_string()
                ));
                st.success = Some(false);
            }
            None => {
                st.error = Some(format!(
                    "avf_framepipe: ffmpeg process could not be reaped; stderr tail:\n{}",
                    stderr_tail.tail_string()
                ));
                st.success = Some(false);
            }
        }
    } else {
        st.success = Some(false);
    }
}

/// Threaded ffmpeg frame writer. Mirrors the ffmpeg argument shape of
/// `StreamingVideoWriter` in `ai/frame_processor.py` (`-y -f rawvideo
/// -vcodec rawvideo -s WxH -r FPS -pix_fmt bgr24 -i - -c:v <codec> [-crf
/// <crf>] [-preset <preset>] -pix_fmt yuv420p <path>`), including that it
/// does *not* set `-movflags faststart` (the Python original doesn't
/// either).
#[pyclass]
struct FrameWriter {
    tx: Option<SyncSender<WriteMsg>>,
    handle: Option<JoinHandle<()>>,
    child: Arc<Mutex<Child>>,
    frames_written: Arc<AtomicU64>,
    state: Arc<Mutex<WriterState>>,
    closed: bool,
}

impl FrameWriter {
    fn check_error(&self) -> PyResult<()> {
        let st = self.state.lock().unwrap();
        if let Some(e) = &st.error {
            return Err(PyRuntimeError::new_err(e.clone()));
        }
        Ok(())
    }

    fn finish_result(&self) -> PyResult<bool> {
        let st = self.state.lock().unwrap();
        if let Some(e) = &st.error {
            return Err(PyRuntimeError::new_err(e.clone()));
        }
        Ok(st.success.unwrap_or(false))
    }
}

#[pymethods]
impl FrameWriter {
    #[new]
    #[pyo3(signature = (
        path, width, height, fps, codec="libx264".to_string(), crf=None, preset=None,
        write_queue=4, ffmpeg_path="ffmpeg".to_string()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        path: String,
        width: usize,
        height: usize,
        fps: f64,
        codec: String,
        crf: Option<i64>,
        preset: Option<String>,
        write_queue: usize,
        ffmpeg_path: String,
    ) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyRuntimeError::new_err(
                "avf_framepipe: width and height must be > 0",
            ));
        }
        let write_queue = write_queue.max(1);

        let mut args: Vec<String> = vec![
            "-y".into(),
            "-f".into(),
            "rawvideo".into(),
            "-vcodec".into(),
            "rawvideo".into(),
            "-s".into(),
            format!("{width}x{height}"),
            "-r".into(),
            format!("{fps}"),
            "-pix_fmt".into(),
            "bgr24".into(),
            "-i".into(),
            "-".into(),
            "-c:v".into(),
            codec,
        ];
        if let Some(c) = crf {
            args.push("-crf".into());
            args.push(c.to_string());
        }
        if let Some(p) = preset {
            args.push("-preset".into());
            args.push(p);
        }
        args.push("-pix_fmt".into());
        args.push("yuv420p".into());
        args.push(path);

        let mut child = Command::new(&ffmpeg_path)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "avf_framepipe: failed to spawn ffmpeg ({ffmpeg_path}): {e}"
                ))
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            PyRuntimeError::new_err("avf_framepipe: failed to capture ffmpeg stdin")
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            PyRuntimeError::new_err("avf_framepipe: failed to capture ffmpeg stderr")
        })?;
        let stderr_tail = StderrTail::spawn(stderr);

        let child = Arc::new(Mutex::new(child));
        let (tx, rx) = sync_channel::<WriteMsg>(write_queue);
        let frames_written = Arc::new(AtomicU64::new(0));
        let state = Arc::new(Mutex::new(WriterState {
            error: None,
            success: None,
        }));

        let thread_child = child.clone();
        let thread_frames_written = frames_written.clone();
        let thread_state = state.clone();
        let handle = thread::spawn(move || {
            writer_thread_body(
                stdin,
                stderr_tail,
                thread_child,
                rx,
                thread_frames_written,
                thread_state,
            );
        });

        Ok(FrameWriter {
            tx: Some(tx),
            handle: Some(handle),
            child,
            frames_written,
            state,
            closed: false,
        })
    }

    /// Enqueue a batch of frames (BGR uint8, shape (h, w, 3)) for writing.
    /// Raises `RuntimeError` immediately if the encoder pipe has already
    /// failed -- never silently accepts writes into a dead pipe.
    fn write_batch<'py>(
        &mut self,
        py: Python<'py>,
        frames: Vec<PyReadonlyArray3<'py, u8>>,
    ) -> PyResult<()> {
        if self.closed {
            return Err(PyRuntimeError::new_err(
                "avf_framepipe: write_batch() called after close()",
            ));
        }
        self.check_error()?;
        if frames.is_empty() {
            return Ok(());
        }

        let mut owned: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
        for arr in &frames {
            let slice = arr.as_slice().map_err(|_| {
                PyRuntimeError::new_err(
                    "avf_framepipe: frame array must be C-contiguous (h, w, 3) uint8",
                )
            })?;
            owned.push(slice.to_vec());
        }

        let tx = self.tx.as_ref().unwrap().clone();
        let send_result = py.detach(|| tx.send(WriteMsg::Batch(owned)));
        if send_result.is_err() {
            self.check_error()?;
            return Err(PyRuntimeError::new_err(
                "avf_framepipe: writer thread ended unexpectedly",
            ));
        }
        Ok(())
    }

    fn frames_written(&self) -> u64 {
        self.frames_written.load(Ordering::SeqCst)
    }

    /// Idempotent: finishes writing, waits for ffmpeg to exit, and returns
    /// whether it exited cleanly. Raises `RuntimeError` (with an ffmpeg
    /// stderr tail) if the pipe failed at any point.
    fn close(&mut self, py: Python<'_>) -> PyResult<bool> {
        if self.closed {
            return self.finish_result();
        }
        self.closed = true;
        self.tx.take();
        let handle = self.handle.take();
        py.detach(|| {
            if let Some(h) = handle {
                let _ = h.join();
            }
        });
        self.finish_result()
    }
}

impl Drop for FrameWriter {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        self.tx.take();
        force_kill_and_reap(&self.child);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[pymodule]
fn avf_framepipe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FrameReader>()?;
    m.add_class::<FrameWriter>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyUntypedArrayMethods;
    use std::io::Cursor;
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
        p.push(format!("avf_framepipe_test_{pid}_{nanos}_{name}"));
        p
    }

    fn ffmpeg_available() -> bool {
        StdCommand::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Generate a tiny synthetic clip via ffmpeg's `testsrc2` lavfi source:
    /// exactly `frames` frames of `w`x`h`, constant frame rate.
    fn make_testsrc_clip(w: u32, h: u32, frames: u32) -> PathBuf {
        let path = unique_temp_path("testsrc.mp4");
        let status = StdCommand::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size={w}x{h}:rate=5"),
                "-frames:v",
                &frames.to_string(),
                "-pix_fmt",
                "yuv420p",
                path.to_str().unwrap(),
            ])
            .status()
            .expect("failed to run ffmpeg to build test fixture");
        assert!(status.success(), "ffmpeg fixture generation failed");
        path
    }

    /// Generate a clip of `colors.len()` frames, each a solid, distinct
    /// color: one single-frame lavfi `color` clip per color, joined via the
    /// concat demuxer (re-encoded, lossless, so frame count is exact --
    /// `-filter_complex concat` on single-frame lavfi sources was observed
    /// to silently drop the last frame in some ffmpeg builds).
    fn make_solid_color_clip(w: u32, h: u32, colors: &[&str]) -> PathBuf {
        let mut parts = Vec::new();
        for c in colors {
            let part = unique_temp_path(&format!("color_{c}.mp4"));
            let status = StdCommand::new("ffmpeg")
                .args([
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    &format!("color=c={c}:s={w}x{h}"),
                    "-frames:v",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    part.to_str().unwrap(),
                ])
                .status()
                .expect("failed to run ffmpeg to build a single-color fixture part");
            assert!(
                status.success(),
                "ffmpeg single-color fixture generation failed"
            );
            parts.push(part);
        }

        let list_path = unique_temp_path("colors_list.txt");
        let list_contents: String = parts
            .iter()
            .map(|p| format!("file '{}'\n", p.to_str().unwrap()))
            .collect();
        std::fs::write(&list_path, list_contents).unwrap();

        let out_path = unique_temp_path("colors.mp4");
        let status = StdCommand::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path.to_str().unwrap(),
                "-c:v",
                "libx264",
                "-crf",
                "0",
                "-pix_fmt",
                "yuv420p",
                out_path.to_str().unwrap(),
            ])
            .status()
            .expect("failed to run ffmpeg to concat color fixture parts");
        assert!(status.success(), "ffmpeg color fixture concat failed");

        for p in &parts {
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_file(&list_path);
        out_path
    }

    // -------------------------------------------------------------
    // read_one_frame: direct, deterministic tests of the frame-boundary
    // vs. mid-frame-truncation distinction (error contract cases 4/5),
    // with no ffmpeg subprocess involved.
    // -------------------------------------------------------------

    #[test]
    fn read_one_frame_reads_a_full_frame() {
        let data = vec![9u8; 10];
        let mut cursor = Cursor::new(data.clone());
        let mut buf = vec![0u8; 10];
        match read_one_frame(&mut cursor, 10, &mut buf) {
            FrameReadOutcome::Full(v) => assert_eq!(v, data),
            _ => panic!("expected Full"),
        }
    }

    #[test]
    fn read_one_frame_detects_clean_eof_at_frame_boundary() {
        let mut cursor = Cursor::new(Vec::<u8>::new());
        let mut buf = vec![0u8; 10];
        match read_one_frame(&mut cursor, 10, &mut buf) {
            FrameReadOutcome::CleanEof => {}
            _ => panic!("expected CleanEof"),
        }
    }

    #[test]
    fn read_one_frame_detects_mid_frame_truncation() {
        let mut cursor = Cursor::new(vec![1u8, 2, 3, 4, 5]); // 5 of 10 bytes
        let mut buf = vec![0u8; 10];
        match read_one_frame(&mut cursor, 10, &mut buf) {
            FrameReadOutcome::Truncated => {}
            _ => panic!("expected Truncated"),
        }
    }

    // -------------------------------------------------------------
    // FrameReader integration tests (spawn real ffmpeg).
    // -------------------------------------------------------------

    #[test]
    fn round_trip_read_known_clip() {
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        let path = make_testsrc_clip(16, 16, 7);
        Python::attach(|py| {
            let mut reader = FrameReader::new(
                path.to_str().unwrap().to_string(),
                16,
                16,
                4,
                2,
                "ffmpeg".to_string(),
            )
            .expect("reader construction failed");
            let mut total = 0usize;
            loop {
                match reader.next_batch(py).expect("next_batch failed") {
                    Some(batch) => {
                        for arr in &batch {
                            let bound = arr.bind(py);
                            assert_eq!(bound.shape(), [16, 16, 3]);
                        }
                        total += batch.len();
                    }
                    None => break,
                }
            }
            assert_eq!(total, 7, "expected exactly 7 decoded frames");
            reader.close(py);
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn no_stale_buffer_aliasing_across_batches() {
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        let colors = ["red", "green", "blue"];
        let path = make_solid_color_clip(8, 8, &colors);
        Python::attach(|py| {
            let mut reader = FrameReader::new(
                path.to_str().unwrap().to_string(),
                8,
                8,
                1,
                3,
                "ffmpeg".to_string(),
            )
            .expect("reader construction failed");

            let mut frames: Vec<Py<PyArray3<u8>>> = Vec::new();
            loop {
                match reader.next_batch(py).expect("next_batch failed") {
                    Some(mut b) => frames.append(&mut b),
                    None => break,
                }
            }
            assert_eq!(frames.len(), 3, "expected exactly 3 decoded frames");

            // Snapshot bytes of every already-returned frame only *after*
            // all frames have been decoded. If the decode thread had
            // handed out views into a single reused buffer (the aliasing
            // bug this test targets) every frame would now show the last
            // decoded color instead of its own.
            let bytes: Vec<Vec<u8>> = frames
                .iter()
                .map(|f| f.bind(py).readonly().as_slice().unwrap().to_vec())
                .collect();
            assert_ne!(
                bytes[0], bytes[1],
                "frame 0 and frame 1 must differ (distinct colors)"
            );
            assert_ne!(
                bytes[1], bytes[2],
                "frame 1 and frame 2 must differ (distinct colors)"
            );
            assert_ne!(
                bytes[0], bytes[2],
                "frame 0 and frame 2 must differ (distinct colors)"
            );

            reader.close(py);
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn backpressure_bounds_frames_read_when_consumer_stalls() {
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        let path = make_testsrc_clip(16, 16, 10);
        Python::attach(|py| {
            let mut reader = FrameReader::new(
                path.to_str().unwrap().to_string(),
                16,
                16,
                1,
                1,
                "ffmpeg".to_string(),
            )
            .expect("reader construction failed");

            // Never call next_batch(); give the decode thread plenty of
            // time to run ahead as far as read_ahead=1 permits.
            thread::sleep(Duration::from_millis(300));
            let stalled = reader.frames_read();
            assert!(
                stalled <= 2,
                "expected read_ahead=1 to bound frames_read() to <=2 while the \
                 consumer never calls next_batch(); got {stalled} (of 10 total frames)"
            );

            // Now drain normally and confirm every frame still arrives.
            let mut total = 0usize;
            loop {
                match reader.next_batch(py).expect("next_batch failed") {
                    Some(b) => total += b.len(),
                    None => break,
                }
            }
            assert_eq!(total, 10);
            reader.close(py);
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reader_errors_on_nonexistent_input_file() {
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        Python::attach(|py| {
            let mut reader = FrameReader::new(
                "/nonexistent/path/does_not_exist.mp4".to_string(),
                16,
                16,
                4,
                2,
                "ffmpeg".to_string(),
            )
            .expect("construction should succeed -- ffmpeg spawns fine, it just fails later");
            let result = reader.next_batch(py);
            assert!(
                result.is_err(),
                "expected a RuntimeError for a nonexistent input file"
            );
            reader.close(py);
        });
    }

    #[test]
    fn reader_errors_on_invalid_ffmpeg_binary() {
        let result = FrameReader::new(
            "/dev/null".to_string(),
            16,
            16,
            4,
            2,
            "/nonexistent/ffmpeg-binary-xyz".to_string(),
        );
        assert!(
            result.is_err(),
            "expected a spawn failure to surface as an error, not a panic or hang"
        );
    }

    #[test]
    fn reader_errors_cleanly_on_truncated_container() {
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        let good = make_testsrc_clip(16, 16, 10);
        let truncated = unique_temp_path("truncated.mp4");
        {
            let bytes = std::fs::read(&good).unwrap();
            let cut = bytes.len() / 3;
            std::fs::write(&truncated, &bytes[..cut]).unwrap();
        }
        Python::attach(|py| {
            let mut reader = FrameReader::new(
                truncated.to_str().unwrap().to_string(),
                16,
                16,
                4,
                2,
                "ffmpeg".to_string(),
            )
            .expect("reader construction failed");
            let mut saw_error = false;
            loop {
                match reader.next_batch(py) {
                    Ok(Some(_)) => continue,
                    Ok(None) => break,
                    Err(_) => {
                        saw_error = true;
                        break;
                    }
                }
            }
            reader.close(py);
            assert!(
                saw_error,
                "expected a truncated/corrupt container to surface as a RuntimeError, \
                 not silent EOS, not a hang, not a panic"
            );
        });
        let _ = std::fs::remove_file(&good);
        let _ = std::fs::remove_file(&truncated);
    }

    // -------------------------------------------------------------
    // FrameWriter integration tests (spawn real ffmpeg).
    // -------------------------------------------------------------

    #[test]
    fn write_read_round_trip() {
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        let path = unique_temp_path("written.mp4");
        Python::attach(|py| {
            let mut writer = FrameWriter::new(
                path.to_str().unwrap().to_string(),
                8,
                8,
                5.0,
                "libx264".to_string(),
                Some(23),
                Some("ultrafast".to_string()),
                4,
                "ffmpeg".to_string(),
            )
            .expect("writer construction failed");

            for _ in 0..6 {
                let arr = vec![128u8; 8 * 8 * 3]
                    .into_pyarray(py)
                    .reshape([8, 8, 3])
                    .unwrap();
                let ro = arr.readonly();
                writer
                    .write_batch(py, vec![ro])
                    .expect("write_batch failed");
            }
            // frames_written() is incremented asynchronously by the writer
            // thread as it actually writes each frame to ffmpeg's stdin, so
            // it can legitimately lag right after write_batch() returns
            // (write_batch() only guarantees the frame was *enqueued*).
            // close() joins the thread, so frames_written() is only
            // meaningful as a final count once it returns.
            let ok = writer
                .close(py)
                .expect("close() reported an ffmpeg failure");
            assert!(ok, "expected ffmpeg to exit cleanly");
            assert_eq!(writer.frames_written(), 6);
        });

        // Round-trip: read it back with FrameReader and confirm 6 frames.
        Python::attach(|py| {
            let mut reader = FrameReader::new(
                path.to_str().unwrap().to_string(),
                8,
                8,
                4,
                2,
                "ffmpeg".to_string(),
            )
            .expect("reader construction failed");
            let mut total = 0usize;
            loop {
                match reader.next_batch(py).expect("next_batch failed") {
                    Some(b) => total += b.len(),
                    None => break,
                }
            }
            assert_eq!(total, 6);
            reader.close(py);
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writer_errors_on_unwritable_output_path() {
        assert!(ffmpeg_available(), "ffmpeg must be on PATH for this test");
        Python::attach(|py| {
            let mut writer = FrameWriter::new(
                "/nonexistent_dir_xyz_avf/out.mp4".to_string(),
                8,
                8,
                5.0,
                "libx264".to_string(),
                None,
                None,
                4,
                "ffmpeg".to_string(),
            )
            .expect("spawn itself should succeed even for a bad output path");

            let mut saw_error = false;
            for _ in 0..5 {
                let arr = vec![64u8; 8 * 8 * 3]
                    .into_pyarray(py)
                    .reshape([8, 8, 3])
                    .unwrap();
                let ro = arr.readonly();
                if writer.write_batch(py, vec![ro]).is_err() {
                    saw_error = true;
                    break;
                }
            }
            let close_result = writer.close(py);
            assert!(
                saw_error || close_result.is_err(),
                "expected an error from write_batch() or close() for an unwritable output path"
            );
        });
    }

    #[test]
    fn writer_errors_on_invalid_ffmpeg_binary() {
        let result = FrameWriter::new(
            "/tmp/avf_framepipe_unused_output.mp4".to_string(),
            8,
            8,
            5.0,
            "libx264".to_string(),
            None,
            None,
            4,
            "/nonexistent/ffmpeg-binary-xyz".to_string(),
        );
        assert!(
            result.is_err(),
            "expected a spawn failure to surface as an error, not a panic or hang"
        );
    }
}
