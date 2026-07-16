"""Auto Video Fixer - Configuration system."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml


def get_config_dir() -> Path:
    """Return platform-appropriate config directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "auto-video-fixer"


def get_data_dir() -> Path:
    """Return platform-appropriate data directory (models, cache)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "auto-video-fixer"


def get_state_dir() -> Path:
    """Return platform-appropriate state directory (run-time logs, etc.).

    Follows the same pattern as get_config_dir()/get_data_dir(): Linux uses
    XDG_STATE_HOME (falling back to ~/.local/state) per the XDG base
    directory spec; macOS/Windows don't have a separate "state" concept in
    their platform conventions, so they reuse the data dir's base (Windows
    LOCALAPPDATA, macOS Application Support) with a "logs" subdirectory
    layered on top by get_log_dir().
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "auto-video-fixer"


def get_log_dir() -> Path:
    """Return the directory automatic per-run log files are written to."""
    return get_state_dir() / "logs"


def prune_old_logs(log_dir: Path, keep: int = 50) -> None:
    """Delete the oldest files in `log_dir` (by mtime) beyond `keep` newest.

    Simple retention for the automatic per-run log file: called once at CLI
    startup, before the current run's log file is created, so it never
    counts (or deletes) the file about to be written.
    """
    if not log_dir.is_dir():
        return
    try:
        files = sorted(
            (p for p in log_dir.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in files[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass


def get_config_path() -> Path:
    """Return path to the main config file."""
    return get_config_dir() / "config.yaml"


_SECRET_KEY_MARKERS = ("api_key", "apikey", "token", "password", "secret")


def _looks_like_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SECRET_KEY_MARKERS)


def redact_secrets(data: Any) -> Any:
    """Return a deep copy of `data` with secret-looking values masked.

    A dict key "looks like a secret" if it contains api_key/apikey/token/
    password/secret (case-insensitive), e.g. "api_key", "API_KEY",
    "openai_api_key". Non-empty values for such keys are replaced with
    "***"; empty/falsy values are left as-is (nothing to leak).
    """
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(k, str) and _looks_like_secret_key(k) and v:
                out[k] = "***"
            else:
                out[k] = redact_secrets(v)
        return out
    if isinstance(data, list):
        return [redact_secrets(v) for v in data]
    return data


def diff_from_defaults(data: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of `data` whose values differ from `defaults`.

    Recurses into nested dicts so only the actually-changed leaf keys show
    up (unchanged sibling keys in a partially-overridden mapping are
    omitted), used to produce a compact "settings that differ from
    DEFAULTS" summary for startup logging.
    """
    diff: dict[str, Any] = {}
    for k, v in data.items():
        default_v = defaults.get(k, _MISSING)
        if isinstance(v, dict) and isinstance(default_v, dict):
            nested = diff_from_defaults(v, default_v)
            if nested:
                diff[k] = nested
        elif default_v is _MISSING or v != default_v:
            diff[k] = v
    return diff


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def deep_merge(base: dict, override: dict, _path: str = "") -> None:
    """Recursively merge `override` onto `base`, in place.

    Shared by ``Config._merge()`` (user config.yaml onto DEFAULTS) and
    ``BaseStage.__init__``'s per-occurrence ``overrides`` param
    (``pipeline.default_order`` entries' ``config:`` key onto the cascaded
    ``stages.<name>`` dict) -- same merge semantics both places: a mapping
    recurses, a scalar overwrites, and a scalar is refused if it would
    clobber a dict-valued default (logged and skipped, not raised) since
    callers unconditionally treat those keys as mappings.
    """
    from autovideofixer.logger import get_logger

    for k, v in override.items():
        key_path = f"{_path}.{k}" if _path else k
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v, key_path)
        elif k in base and isinstance(base[k], dict) and not isinstance(v, dict):
            # Refuse to clobber a default mapping (e.g. stages.upscale, quality.quality_target)
            # with a scalar override -- consumers unconditionally call .get()/.items() on
            # these and would crash with an unhandled AttributeError otherwise.
            get_logger("autovideofixer.config").warning(
                "Ignoring invalid config override for '%s': expected a mapping, got %s",
                key_path,
                type(v).__name__,
            )
        else:
            base[k] = v


class Config:
    """Central configuration manager. Loads from disk and provides defaults."""

    DEFAULTS: dict[str, Any] = {
        "general": {
            "output_dir": None,  # None means same as input
            "output_container": "mp4",  # extension for auto-generated output filenames
            # ("mp4" or ".mp4"); None/empty keeps the input's own extension.
            "temp_dir": None,
            "max_concurrent_jobs": 1,
            "log_level": "INFO",
            "overwrite": False,
            "use_ai": None,  # None=auto (preset/config), True=force AI, False=force traditional
            # Global default for AI-capable stages (upscale, interpolate,
            # denoise_video, deblock) when the AI path can't run (torch
            # missing, model load failure, inference exception, CUDA OOM
            # after tiling retries are exhausted). True (default) = fall
            # back to the stage's traditional FFmpeg implementation with a
            # WARNING log. False = fail the stage instead of silently
            # producing traditional output. Per-stage
            # stages.<name>.ai_fallback overrides this when not null.
            "ai_fallback": True,
        },
        "gpu": {
            "auto_detect": True,
            "preferred_device": "auto",  # auto, cpu, cuda, metal
            "memory_limit_gb": None,
            # Which Vulkan physical device index the ncnn backend uses (stages.<name>.backend:
            # ncnn only -- irrelevant to the torch backend, which uses gpu.preferred_device
            # instead). 0 = ncnn's own default device. On a multi-GPU machine this is NOT
            # necessarily your fastest/discrete GPU -- e.g. an iGPU can enumerate before a
            # passed-through dGPU. Check `avf gpu-info` (or `ncnn.get_gpu_count()`/
            # `ncnn.get_gpu_info(i).device_name()`) to find the right index.
            "vulkan_device": 0,
        },
        "ffmpeg": {
            "binary": None,  # None = auto-detect in PATH
            "hwaccel": "auto",  # auto, cuda, vulkan, qsv, vaapi, none
            "threads": 0,  # 0 = auto
        },
        "quality": {
            "vmaf_model": "vmaf_v0.6.1",
            "vmaf_features": "psnr,ssim,ms_ssim,fast",
            "quality_target": {
                "mode": "none",  # none, min, avg, max
                "target": 95.0,
                "max_loss_pct": 5.0,
                "target_resolution": None,  # (width, height) or None
                "target_framerate": None,
                "keep_aspect_ratio": True,  # preserve original video aspect ratio
            },
        },
        "pipeline": {
            # Drives actual execution order/omission/repetition --
            # Pipeline.resolve_stage_order() (called by optimize_stage_order()/
            # execute_job()) reads this list; the hardcoded DEFAULT_STAGE_ORDER
            # fallback in core/pipeline.py only kicks in if this key is
            # missing/empty. Entries are either a plain stage name (string --
            # behaves exactly as before: runs iff the stage is in the
            # requested/auto-determined set) or a mapping
            # ``{stage, enabled, config}`` for explicit per-occurrence control
            # (force-run/force-skip regardless of global gating, and/or
            # per-occurrence config overrides) and repetition (the same stage
            # name can appear more than once; each occurrence resolves and
            # runs independently). See docs/config.example.yaml for the full
            # mapping-entry syntax and AGENTS.md's "Pipeline Behavior" section
            # for semantics. DEFAULTS itself stays plain strings so
            # out-of-box behavior is unaffected by this feature.
            #
            # deblock now runs BEFORE stabilize (previously the reverse):
            # blocking artifacts come from the source video, so deblocking
            # before stabilization's perspective warping keeps the deblock
            # model's input accurate (no warped block edges) and gives the
            # stabilizer cleaner detail to track motion against.
            "default_order": [
                "detect",
                "deblock",
                "stabilize",
                "crop",
                "denoise_video",
                "upscale",
                "interpolate",
                "normalize_volume",
                "normalize_audio",
                "speed",
                "hdr",
                "encode",
            ],
            # Must stay >= len(default_order) above (12) since a full default run
            # legitimately uses every stage; this only guards against pathological
            # --stage/default_order repetition, not normal preset/auto-determined
            # pipelines. Counts resolved OCCURRENCES, not unique stage names -- a
            # stage repeated via default_order counts once per repetition.
            "max_stages": 15,
            "skip_stage_on_error": True,
        },
        "stages": {
            "upscale": {
                "enabled": True,
                "ai_model": "RealESRGAN_x4plus",
                "traditional_method": "superres",
                "scale_factor": 4,
                "tta_mode": 0,
                # 0 = auto: tile only when a frame's resolution risks CUDA OOM
                # (see RealESRGANUpscaler.AUTO_TILE_THRESHOLD_PX) or on a
                # caught OOM. Set > 0 to always tile at that pixel size.
                "tile_size": 0,
                # Frames batched into one forward pass in RealESRGANUpscaler.
                # upscale_video() (whole-*frame* batching -- N different
                # frames per call). 1 (default) = today's one-frame-at-a-time
                # behavior; opt-in until field-tested. Only takes effect for
                # frames small enough to skip tiling -- see tile_batch_size
                # below for the batching that matters at e.g. 4K, where every
                # frame always tiles regardless of this setting.
                "batch_size": 1,
                # Tiles (sharing the same padded input shape) batched into one
                # forward pass inside run_tiled_inference(), when a frame's
                # resolution triggers tiled inference (see tile_size/
                # AUTO_TILE_THRESHOLD_PX above). 1 (default) = today's
                # one-tile-at-a-time behavior. This is the batching knob that
                # matters for large frames -- a 4K frame at the default
                # tile_size=512 needs a 5x8=40-tile grid, so raising this
                # collapses many small sequential forward passes into a
                # handful of larger ones. Independent of batch_size above
                # (different batching axis: N tiles of ONE frame, vs N whole
                # frames) -- opt-in for the same reason.
                "tile_batch_size": 1,
                # None = inherit general.ai_fallback; True/False overrides it
                # for this stage only.
                "ai_fallback": None,
                # Inference backend: "torch" (PyTorch/CUDA) or "ncnn" (Vulkan via
                # the ncnn Python package -- portable to AMD/Intel/iGPUs). Falls
                # through the ai_fallback policy if the backend is unavailable.
                "backend": "torch",
                # CRF for the AI stage's internal temp encode (StreamingVideoWriter/
                # frames_to_video), NOT the final output. Default 16 (near-visually-
                # lossless) rather than libx264's own default (23) -- the temp file
                # used to be encoded at CRF 23 and then RE-encoded by the stage's mux
                # pass at CRF 18, a double lossy generation plus a wasted full x264
                # pass; the mux pass now stream-copies (`-c:v copy`) instead, so this
                # is the ONLY encode the AI-processed frames actually go through.
                "temp_crf": 16,
                # Frame-transport backpressure knobs (ai/frame_pipe.py) --
                # only meaningfully vary the Rust (avf_framepipe) backend's
                # bounded channels; the pure-Python fallback's writer queue
                # depth is a fixed constant in frame_processor.py (see
                # ai/frame_pipe.py's "fallback parity gaps" docstring note).
                # read_ahead: max decoded chunks buffered ahead of inference.
                "read_ahead": 2,
                # write_queue_depth: max encoded chunks buffered ahead of the
                # ffmpeg encoder pipe.
                "write_queue_depth": 4,
            },
            "interpolate": {
                "enabled": True,
                "ai_model": "rife_v4.6",
                "traditional_method": "minterpolate",
                "ai_fallback": None,  # see "upscale".ai_fallback above
                # "torch" | "ncnn" -- see "upscale".backend. RIFE's ncnn backend
                # uses the separate "rife-ncnn-vulkan-python" package (the
                # generic ncnn Python bindings lack RIFE's custom rife.Warp
                # layer); requires the "ncnn" extra installed, otherwise falls
                # back per ai_fallback policy.
                "backend": "torch",
                # Traditional (minterpolate) path only -- the AI/RIFE path stays serial
                # (GPU-bound; concurrent GPU jobs contend rather than speed things up).
                # minterpolate is single-threaded per ffmpeg process, so a long clip is
                # split into N time-chunks (1-frame overlap trimmed on concat) and run as
                # N parallel ffmpeg processes, then concatenated back together.
                # 0 = auto (min(os.cpu_count(), 8)); 1 = disable chunking (previous serial
                # behavior, one ffmpeg process for the whole input).
                "parallel_chunks": 0,
                # Minimum chunk length in seconds -- below this, chunking isn't worth the
                # per-chunk ffmpeg startup/concat overhead and the input runs as one chunk.
                "min_chunk_duration_sec": 5.0,
                "temp_crf": 16,  # see "upscale".temp_crf above; only used by the AI/RIFE path
            },
            "denoise_video": {
                "enabled": True,
                "ai_model": "RealESRGAN_x4plus",
                "traditional_method": "hqdn3d",
                "tile_size": 0,  # see "upscale".tile_size above
                "batch_size": 1,  # see "upscale".batch_size above
                "tile_batch_size": 1,  # see "upscale".tile_batch_size above
                "ai_fallback": None,  # see "upscale".ai_fallback above
                "temp_crf": 16,  # see "upscale".temp_crf above
                "read_ahead": 2,  # see "upscale".read_ahead above
                "write_queue_depth": 4,  # see "upscale".write_queue_depth above
            },
            "denoise_audio": {
                "enabled": True,
                "ai_model": "demucs",
                "traditional_method": "afftdn",
            },
            "deblock": {
                "enabled": True,
                "strength": "medium",  # low, medium, high
                "tile_size": 0,  # see "upscale".tile_size above
                "batch_size": 1,  # see "upscale".batch_size above
                "tile_batch_size": 1,  # see "upscale".tile_batch_size above
                "ai_fallback": None,  # see "upscale".ai_fallback above
                "temp_crf": 16,  # see "upscale".temp_crf above
                "read_ahead": 2,  # see "upscale".read_ahead above
                "write_queue_depth": 4,  # see "upscale".write_queue_depth above
            },
            "stabilize": {
                "enabled": True,
                "threshold": 2.0,  # stabilize if shake > this value
                "smoothness": 40,  # frames for lowpass filtering (higher = smoother)
                "maxshift": 20,  # max pixels to shift per frame (limits overcorrection)
                "optalgo": "gauss",  # optimization algorithm (opt, gauss, avg)
                "shakiness": 10,  # motion detection sensitivity (1-10, higher = more sensitive)
                "zoom_enabled": True,  # auto zoom-out for very shaky video
                "zoom_threshold": 50.0,  # min movement (px) to trigger zoom
                "zoom_mode": "black",  # black or keep
                "sharpen_enabled": True,  # auto sharpen after stabilization
                # How much of the clip should end up border-free once zoom_enabled's
                # gate decides zoom applies at all (0.0-1.0). 1.0 (default) = today's
                # behavior: vidstabtransform's own optzoom=1 ("optimal static zoom"),
                # sized to the single worst frame so NO frame ever shows a border.
                # 0.0 = no zoom at all (every border from camera motion stays visible).
                # In between: a static zoom= percentage is computed from the
                # zoom_coverage-quantile of per-frame required-zoom estimates (see
                # StabilizeStage._compute_static_zoom_pct) instead of the max --
                # trades "guaranteed no border, ever" for "less aggressive crop,
                # occasional brief borders on the most extreme motion". See AGENTS.md's
                # stabilize zoom section for the accuracy caveat (approximates the
                # smoothed camera path from raw per-frame local-motion values).
                "zoom_coverage": 1.0,
            },
            "normalize_volume": {
                "enabled": True,
                "target_db": -23.0,  # EBU R128
                "true_peak_db": -2.0,
            },
            "hdr_to_sdr": {
                "enabled": False,
                "method": "bt2020",
            },
            "speed": {
                "enabled": False,
                "factor": 1.0,
            },
            "crop": {
                # Auto-crop: detect and remove black borders (letterbox/pillarbox),
                # including residual borders stabilize can introduce. Strictly
                # opt-in -- see docs/REQUIREMENTS.md feature 3, AGENTS.md stage list.
                "enabled": False,
                # cropdetect luma threshold (0-255, ffmpeg default is 24) -- pixels
                # darker than this are treated as "black" border.
                "limit": 24,
                # cropdetect round=2 keeps cropped width/height even (H.264
                # requirement), matching UpscaleStage._round_to_even's convention.
                "round": 2,
                # Skip cropping entirely if the detected crop would save fewer than
                # this many pixels in BOTH width and height -- avoids a pointless
                # 2px crop from encoder rounding noise.
                "min_crop_px": 8,
                # 0 = scan the whole video (reset=0 accumulates the tightest safe
                # crop across every frame scanned, i.e. the furthest real content
                # ever reaches toward each edge -- a per-frame crop would flicker).
                # >0 = seconds to sample from the start of the video instead, for
                # very long inputs where a full scan is too slow.
                "analyze_duration_sec": 0,
                # Optional VLM-assisted disambiguation: a watermark/logo sitting
                # outside the true content area (e.g. positioned relative to a
                # letterboxed frame) can fool naive cropdetect into "protecting" it
                # as non-black content. When enabled AND analysis.vlm.enabled is
                # true, one frame is rendered twice (plain + the proposed crop box
                # drawn via drawbox) and sent to the VLM with a narrow fixed
                # question. Off by default; fails open (WARNING + proceed with the
                # plain cropdetect result) on any VLM error.
                "vlm_check": False,
                # "warn" (default): log the VLM's objection but still crop.
                # "skip": don't crop this video at all if the VLM flags content
                # outside the box. "expand" (growing the crop box to include the
                # flagged region) was considered and rejected -- VLMs don't return
                # reliable pixel coordinates, so there's nothing to expand *to*.
                "vlm_policy": "warn",
            },
        },
        "scenes": {
            # Master switch for scene-based processing (see docs/REQUIREMENTS.md features
            # 1-2, AGENTS.md "Scene mode"). Off by default -- zero behavior change to the
            # existing whole-video pipeline when disabled. When enabled, stabilize and
            # interpolate (if requested for the job) run per-scene instead of on the whole
            # file, then scenes are concatenated back together before the remaining
            # whole-video stages (upscale/denoise/deblock/normalize/encode) run.
            "enabled": False,
            # Reuses analysis.event_detection.scene_change_threshold/min_scene_duration_sec
            # for boundary detection -- no separate scene-mode threshold.
            "drop_non_content": False,  # see analysis.llm below; requires VLM + coordinator
            # Re-encode codec/crf used for the per-scene split/concat intermediate passes
            # (segments get re-encoded again by the final whole-video "encode" stage, so
            # this only needs to be visually lossless-ish, not archival quality).
            "intermediate_crf": 14,
            "intermediate_preset": "veryfast",
            "stabilize": {
                # Whether per-scene stabilize-strength tiering runs at all when scene mode
                # is active and the "stabilize" stage was requested for the job. If false,
                # stabilize still runs per-scene (never across a cut) but with the stage's
                # normal (non-tiered) config for every scene.
                "enabled": True,
                # avg_shake (px, from StabilizeStage's own TRF analysis) above which a
                # scene is escalated from "normal" to "aggressive" tier (re-run with
                # smoothness multiplied by aggressive_smoothness_multiplier). Below the
                # stage's own stabilize.threshold, a scene is already skipped entirely
                # ("skip" tier) by StabilizeStage's existing needs_stab logic.
                "aggressive_shake_threshold": 8.0,
                "aggressive_smoothness_multiplier": 2.0,
            },
        },
        "analysis": {
            "llm": {
                # Coordinating text-LLM used only when scenes.drop_non_content is true.
                # Reviews ALL per-scene VLM summaries together and returns which scene
                # indices (if any) are not part of the video's main content and should be
                # dropped (e.g. a "like and subscribe" interstitial, a channel bumper, an
                # unrelated promotional insert). Can be a cheaper/faster text-only model
                # than analysis.vlm's vision model -- separate provider/model/url/key.
                # Same provider set and the same allow_http/HTTPS gate as analysis.vlm.
                "provider": "ollama",  # ollama, openai, api
                "model": "llama3",
                "api_key": "",
                "api_url": "",
                "allow_http": False,
                # Response token budget (max_tokens / Ollama num_predict) for the
                # coordinator call. Reasoning models spend "thinking" tokens from
                # this same budget before emitting their JSON -- keep this
                # generous or the response gets truncated mid-thought (which
                # fails open: no scenes dropped).
                "max_tokens": 4096,
            },
            "vlm": {
                "enabled": False,
                "provider": "local",  # local, api, ollama, openai
                "model": "llava",
                "api_key": "",
                "api_url": "",
                # Plain-HTTP api_url endpoints are refused for non-loopback
                # hosts unless this is true (frames + api_key travel
                # unencrypted; only enable for a server on a network you
                # control, e.g. a LAN inference box).
                "allow_http": False,
                # Response token budget (max_tokens / Ollama num_predict) per VLM
                # call. Raise if using a reasoning VLM whose thinking tokens
                # count against the response budget.
                "max_tokens": 1024,
                "max_sample_frames": 8,
                "sample_interval_sec": 10.0,
                # Extra text appended to the default (or overridden) user prompt when
                # non-empty, e.g. job-specific context like "these are wildlife clips
                # from a trail camera". Also settable per-run via `avf analyze
                # --prompt-append`.
                "prompt_append": "",
                # Replaces the default user prompt entirely when non-empty.
                # `prompt_append` (if set) is still appended after this. Also settable
                # per-run via `avf analyze --prompt-override`. WARNING: the default
                # prompt instructs the model to return JSON with summary/tags/objects/
                # rating -- an override that changes the requested output format will
                # break that JSON parsing (see _parse_vlm_response's fallback below).
                "prompt_override": "",
                # Replaces the default system prompt entirely when non-empty. No CLI
                # flag; config-only.
                "system_prompt_override": "",
            },
            "event_detection": {
                "enabled": True,
                # Frame-differencing sensitivity (mean fractional luma change between
                # consecutive downscaled frames, 0-1; see _detect_scene_changes()'s
                # docstring in core/analysis.py for the full metric writeup and
                # calibration data). 0.3 (the old default) under-detected real cuts;
                # 0.15 was calibrated against a synthetic ground-truth clip to catch
                # every hard cut with zero false positives. Also settable per-run via
                # `avf analyze --scene-threshold`.
                "scene_change_threshold": 0.15,
                # Cuts closer together than this are merged into the following scene
                # rather than starting a new one -- doesn't affect cut *detection*,
                # only whether a short segment gets its own SceneEvent. Also settable
                # per-run via `avf analyze --min-scene-duration`.
                "min_scene_duration_sec": 2.0,
                "classify_events": False,
            },
            "duplicate_detection": {
                "enabled": True,
                "similarity_threshold": 0.85,
                # `hash_type` was removed (R5.2): the old ahash/dhash/combined choice
                # no longer applies now that compute_video_phash() always uses a
                # single pHash (DCT-based) algorithm -- see core/analysis.py and
                # rust/avf_hashing/ for why pHash replaced both.
            },
        },
    }

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        config_path: str | Path | None = None,
        require_exists: bool = False,
    ):
        """Create a Config.

        Args:
            path: Explicit config file path (positional, pre-existing API).
            config_path: Same as `path`, accepted as a keyword alias so
                callers that pass an explicit path (e.g. the CLI's
                `--config`/`AVF_CONFIG`) can be self-documenting about intent.
                `path` wins if both are given.
            require_exists: If True, the resolved path must already exist on
                disk -- raises FileNotFoundError instead of silently falling
                back to DEFAULTS. Used for an explicitly-requested config
                path (CLI flag / env var), where a typo'd path should be a
                hard error, not a silent no-op. The default platform config
                path (neither `path` nor `config_path` given) is never
                required to exist.
        """
        resolved = path or config_path
        self._path = Path(resolved) if resolved is not None else get_config_path()
        self._require_exists = require_exists and resolved is not None
        if self._require_exists and not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")
        self._data = self._merge()
        self._save_pending = False

    def _merge(self) -> dict[str, Any]:
        merged = self._deep_copy(self.DEFAULTS)
        if self._path.exists():
            try:
                with open(self._path) as f:
                    user = yaml.safe_load(f) or {}
                self._deep_update(merged, user)
            except Exception as e:
                from autovideofixer.logger import get_logger

                get_logger("autovideofixer.config").warning(
                    "Failed to load config from %s: %s. Using defaults.", self._path, e
                )
        return merged

    @staticmethod
    def _deep_copy(d: dict) -> dict:
        import copy

        return copy.deepcopy(d)

    @staticmethod
    def _deep_update(base: dict, override: dict, _path: str = "") -> None:
        """Recursively merge `override` onto `base`, in place.

        Thin backward-compatible wrapper -- the actual merge logic is the
        module-level `deep_merge()`, reused by BaseStage.__init__ for
        per-occurrence stage config overrides (see pipeline.default_order's
        `config:` key).
        """
        deep_merge(base, override, _path)

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node

    def set(self, value: Any, *keys: str) -> None:
        node = self._data
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            elif not isinstance(node[k], dict):
                raise TypeError(
                    f"Cannot set config key {'.'.join(keys)!r}: "
                    f"{'.'.join(keys[: keys.index(k) + 1])!r} is a {type(node[k]).__name__}, "
                    "not a mapping"
                )
            node = node[k]
        node[keys[-1]] = value
        self._save_pending = True

    def save(self) -> None:
        if not self._save_pending:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)
        self._save_pending = False

    @property
    def data(self) -> dict[str, Any]:
        return self._data
