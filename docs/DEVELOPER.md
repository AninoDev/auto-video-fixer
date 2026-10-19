# Auto Video Fixer - Developer Documentation

## Architecture Overview

Auto Video Fixer is a modular video processing application built on Python with Qt GUI and CLI interfaces. The architecture follows a pipeline pattern where processing stages are composed into ordered workflows.

## Core Components

### 1. Pipeline Engine (`core/pipeline.py`)

The pipeline orchestrates processing stages in optimal order:

- **Job Queue**: Manages pending, running, and completed jobs
- **Stage Ordering**: Automatically determines optimal processing order
- **Resource Management**: Handles GPU allocation and concurrent processing
- **Quality Estimation**: Uses VMAF to verify output quality

### 2. Processing Stages (`core/stages/`)

Each stage is a modular, independently testable unit:

```python
class BaseStage(ABC):
    name: str                    # Unique identifier
    category: str                # analysis, enhancement, encoding, output
    priority: int                # Execution order (lower = earlier)
    supports_gpu: bool           # Can use GPU acceleration
    supports_hardware_encoding: bool  # Can use HW encoding
    
    def should_run(self, input_info) -> tuple[bool, str | None]:
        """Determine if stage should execute"""
        
    def execute(self, input_path, output_path, progress_callback, **kwargs) -> StageResult:
        """Execute the processing"""
```

### 3. Quality Estimation (`core/quality.py`)

VMAF (Video Multi-Method Assessment Fusion) provides perceptual quality metrics:

- **VMAF Score**: 0-100 (higher = better perceptual quality)
- **PSNR/SSIM**: Traditional metrics for comparison
- **Quality Modes**: min, avg, max, target-based filtering

### 4. Video Analysis (`core/analysis.py`)

Intelligent video understanding:

- **Scene Detection**: Frame differencing for scene boundaries
- **VLM Integration**: Vision Language Models for content understanding
- **Duplicate Detection**: Perceptual hashing for similarity matching

### 5. Configuration System (`config.py`)

Platform-aware configuration with YAML storage:

- Automatic detection of FFmpeg, GPU, and system capabilities
- Preset-based configuration with user overrides
- Cross-platform path handling

## Processing Pipeline Order

The optimal processing order is:

1. **detect** - Analyze input properties (the full order, including `retime`, `crop`, the opt-in
   `analyze` after `crop` and `scene_subtitles` / `analysis_sidecar` before `encode`, is
   `pipeline.default_order`)
2. **stabilize** - Correct camera shake (before enhancement)
3. **deblock** - Remove compression artifacts
4. **denoise_video** - Reduce noise (before upscaling)
5. **upscale** - Increase resolution
6. **interpolate** - Increase framerate
7. **normalize_volume** / **normalize_audio** - Audio normalization
8. **speed** - Speed adjustment
9. **hdr** - HDR conversion
10. **encode** - Final encoding

## Adding Custom Stages

### Step 1: Create Stage File

```python
# src/autovideofixer/core/stages/my_stage.py
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus

class MyStage(BaseStage):
    name = "my_stage"
    display_name = "My Custom Stage"
    description = "Description of what this stage does"
    category = "enhancement"
    priority = 25
    supports_gpu = True
    
    def __init__(self, config):
        super().__init__(config)
        self._stage_config = config.get("stages", self.name, default={})
```

### Step 2: Implement Logic

```python
def should_run(self, input_info):
    if not self.is_enabled():
        return False, "Disabled"
    return True, None

def execute(self, input_path, output_path, progress_callback, **kwargs):
    import time
    start = time.time()
    
    self._report_progress(0.0, "Processing...", progress_callback)
    
    # Your processing logic here
    # Use FFmpeg, PyTorch, OpenCV, etc.
    
    self._report_progress(1.0, "Done", progress_callback)
    
    return StageResult(
        status=StageStatus.COMPLETED,
        output_path=output_path,
        duration_sec=time.time() - start,
    )
```

### Step 3: Register Stage

Add to `src/autovideofixer/core/stages/__init__.py`:

```python
from autovideofixer.core.stages.my_stage import MyStage

# Register as a bare function call (NOT a decorator)
register_stage(MyStage)
```

**Note**: `register_stage()` works as both a decorator (`@register_stage`) and a bare function call (`register_stage(MyStage)`). Both are valid.

### Step 4: Add Stage Config

Add to `DEFAULTS["stages"]` in `src/autovideofixer/config.py`:

```python
"stages": {
    "my_stage": {
        "enabled": True,
        # ... stage-specific settings
    },
}
```

### Step 5: Set Stage Order

**CRITICAL**: The pipeline does **not** use `DEFAULTS["pipeline"]["default_order"]`. The actual ordering is hardcoded in `Pipeline.optimize_stage_order()` in `pipeline.py:232`. To change order, modify that method.

## Writing a stage that uses analysis

(REQUIREMENTS.md § 28.) The reference implementations are `core/stages/analysis_sidecar.py` (about
100 lines) and `core/stages/scene_subtitles.py`; this walks through what they do.

### 1. The facade

```python
from autovideofixer.core.analysis_service import analyze_video

result = analyze_video(path, config, profile="thorough", scope="scenes", cache=self.analysis_cache)
doc = result.to_dict()        # JSON-serialisable: scenes[], vlm, vlm_calls, scope, profile, ...
for sc in doc["scenes"]:
    sc["start_time"], sc["end_time"], (sc.get("vlm") or {}).get("summary")
```

`analyze_video` applies the profile, validates (`ValueError` for an unknown profile / bad analysis
config, before any work), runs scene detection and the VLM per `scope`, and returns an
`AnalysisResult` (`.analysis` is the underlying `VideoAnalysis`). A VLM that is down is **not** an
exception: the VLM fields are empty and the failed calls are in `vlm_calls`. Per-scene descriptions
exist only when the scope includes scenes *and* the VLM is on. Keyword `overrides=` takes the
`VideoAnalyzer.analyze` per-call overrides (`OVERRIDE_KEYS`).

### 2. The cache

The pipeline creates one `AnalysisCache` per job and sets `stage.analysis_cache` on every stage
instance. Pass it as `cache=`: a call with the same file (realpath + size + mtime) and the same
effective analysis config (profile, scope, overrides included) returns the earlier result
(`result.cached` is True, no new VLM calls, the `VideoAnalysis` object is shared -- do not mutate
it). Thread-safe; concurrent callers compute once. There is deliberately no module-level cache.
Scripts outside the pipeline can make their own `AnalysisCache()`.

### 3. `input_info["analysis"]`

If the `analyze` stage ran, `input_info["analysis"]` already holds `to_dict()` and
`input_info["analysis_time_scale"]` the timeline factor (below). Use the helper rather than
re-implementing the choice:

```python
got = obtain_analysis(self.config, self.name, self._stage_config, input_path, input_info,
                      self.analysis_cache, need_scenes=True)
if got is None:       # no analysis and not allowed / able to make one: it already warned once
    return StageResult(status=StageStatus.SKIPPED, skipped_reason="no analysis available")
analysis, time_scale = got
```

`obtain_analysis` prefers the published analysis (with `need_scenes=True`, only if it has per-scene
results, i.e. scope `scenes`/`both`; a `video`-scope publication falls through. Better: have
`analyze` ask for what you need up front -- it reads `input_info["planned_stages"]`, the job's
stage names in run order, and widens its scope for a later `scene_subtitles`; add your stage
there if it needs scenes); otherwise, with `stages.<name>.analyze_if_missing`
(default true) it runs `analyze_video` itself through the cache (so two writers share one run);
otherwise it warns and returns None. **A stage that consumes analysis must fail open**: skip with one
WARNING, never FAILED, unless you want a missing analysis to fail the job.

### 4. Timestamp mapping

Analysis times are in the timeline of the file that was analysed -- the intermediate at the
`analyze` stage's position. `retime` keeps real timestamps and `interpolate` keeps the duration, so
only a stage that rescales time changes them (`speed`). Such a stage reports `metadata["time_scale"]`
(output seconds per input second; `speed` reports `1 / factor`) and the pipeline multiplies
`input_info["analysis_time_scale"]` by it. Convert with `map_time(t, input_info)` (or multiply by the
`time_scale` `obtain_analysis` returned), and clamp to `input_info["duration"]`. An analysis you ran
yourself on your own input has scale 1.0. If you add a stage that changes the timeline, report
`time_scale`; otherwise consumers downstream will be wrong.

### 5. Where the stage goes, and where its output goes

`encode` is always the last stage and every earlier stage is handed a temp path, so a stage that
writes *beside the deliverable* takes the destination from `input_info["final_output_path"]` (injected
by the pipeline, carried across re-probes). Position it with `pipeline.default_order`:

- needs the picture before the final encode (`burn`) or must not be sped up: after `speed`/`hdr`, before `encode`;
- sidecar / stream-copy mux (`soft`): same slot -- `encode` is the one that writes the file, and it
  copies the muxed text track through (`-c:s` comes from `input_info["soft_subtitle_codec"]`);
- only consumes analysis (`analyze` itself): early (after `crop`, before the heavy stages).

Rules to copy: a stage that writes no media file returns **no** `output_path` (never
`input_path` -- the pipeline's temp cleanup deletes recorded outputs, and it once deleted a job's
source this way) and, if `produces_output` is True for its other modes, sets
`self.produces_output = False` on the instance. Do not clobber silently: follow `general.overwrite`
for files beside the output. Add `-fps_mode passthrough` (`timing_output_args()`) to any ffmpeg step
that re-encodes; stream copy needs nothing. Filter options that embed a path need
`scene_subtitles.escape_filter_value` (two escaping levels).

### 6. Registration, config, order

Same as any stage (see "Adding Custom Stages"): subclass `BaseStage`, `register_stage(MyStage)` in
`core/stages/__init__.py`, defaults in `Config.DEFAULTS["stages"][name]` (`enabled: False` for an
opt-in stage), add the name to `pipeline.default_order` **and** `DEFAULT_STAGE_ORDER` in
`core/pipeline.py`, raise `pipeline.max_stages`, mirror all of it in both example configs
(byte-identical), and -- because a bootstrapped user config pins `default_order` -- add a
`config_migrations.py` step that inserts the stage and bump `CURRENT_CONFIG_VERSION`. If the stage
should be planned when enabled, add it to `Pipeline.auto_determine_stages`.

### 7. Testing

- Fake the seams, not the stage: patch `autovideofixer.core.analysis._detect_scene_changes`,
  `VideoAnalyzer.run_vlm_analysis` / `run_vlm_analysis_for_scene` (see `tests/unit/test_analysis_service.py`),
  or hand the stage a ready `input_info["analysis"]` (see `test_analysis_stages.py`).
- Assert the ffmpeg argument list with `run_ffmpeg` mocked for the degrade paths, and run real ffmpeg
  in `tests/integration/test_analysis_stages_e2e.py`-style tests through `Pipeline.execute_job` with a
  per-test `Config`: check `ffprobe` streams, durations and, for burn-in, a frame diff.
- Test the timeline: a real run with `stages.speed.factor != 1` after `analyze`, compare your times
  with the output's duration.
- Run any ad-hoc `avf` command with all four `XDG_*` variables pointing at a scratch dir.

External (out-of-package) stage plugins are not supported yet; a stage has to live in
`core/stages/` and be registered there.

## GPU Integration

### PyTorch Models

```python
import torch

class MyAIStage(BaseStage):
    def execute(self, input_path, output_path, progress_callback, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyModel().to(device)
        
        # Process frames on GPU
        for frame in frames:
            frame_tensor = frame.to(device)
            output = model(frame_tensor)
```

### FFmpeg Hardware Acceleration

```python
from autovideofixer.core.ffmpeg_utils import resolve_hwaccel, build_hwaccel_args

hwaccel = resolve_hwaccel("auto")  # Auto-detect best method
hw_args = build_hwaccel_args(hwaccel)  # [-hwaccel, cuda]

# Use in FFmpeg command
args = hw_args + ["-i", input_path, "-c:v", "h264_nvenc", output_path]
```

## Quality Control with VMAF

```python
from autovideofixer.core.quality import estimate_quality_vmaf, QualityMode

# Estimate quality
result = estimate_quality_vmaf(
    reference="original.mp4",
    distorted="processed.mp4",
    model="vmaf_v0.6.1",
)

# Check if quality meets target
if result.mode == QualityMode.TARGET:
    if result.vmaf_score >= 95.0:
        print("Quality acceptable")
    else:
        print("Quality too low")
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipeline.py

# Run with verbose output
pytest -v
```

## Debugging

Enable debug logging:

```python
from autovideofixer.logger import setup_logging
setup_logging("DEBUG")
```

Or via CLI:

```bash
avf --verbose process video.mp4
# or
avf --log-level DEBUG process video.mp4
```

## Performance Tips

1. **Batch Processing**: Process multiple files concurrently
2. **GPU Acceleration**: Use hardware encoding when available
3. **Skip Unnecessary Stages**: Let `should_run()` determine what's needed
4. **Intermediate Files**: Use temp files for multi-stage processing
5. **Progress Reporting**: Use `_report_progress()` for UI responsiveness

## File Structure

```
src/autovideofixer/
├── __init__.py              # Package metadata
├── config.py                # Configuration management
├── logger.py                # Logging setup
├── core/
│   ├── pipeline.py          # Pipeline orchestrator
│   ├── quality.py           # VMAF quality estimation
│   ├── analysis.py          # Video analysis utilities
│   ├── ffmpeg_utils.py      # FFmpeg integration
│   ├── presets.py           # Processing presets
│   └── stages/              # Processing stages
│       ├── base.py          # Base class + registry
│       ├── detect.py        # Analysis stage
│       ├── stabilize.py     # Stabilization
│       ├── deblock.py       # Deblocking
│       ├── denoise_video.py # Video denoising
│       ├── upscale.py       # Resolution upscaling
│       ├── interpolate.py   # Frame interpolation
│       ├── normalize_audio.py # Audio normalization
│       ├── encode.py        # Video encoding
│       ├── remux.py         # Container remuxing
│       ├── speed.py         # Speed adjustment
│       └── hdr.py           # HDR conversion
├── gui/
│   └── main_window.py       # Qt GUI
└── cli/
    └── cli.py               # Click CLI
```

## Dependencies

### Core
- **FFmpeg**: Video processing backend
- **PyYAML**: Configuration file parsing

### AI/ML (`ai/`)

The AI package provides PyTorch-based AI processing with graceful fallback to traditional FFmpeg methods:

- **`torch_utils.py`** - PyTorch device detection, tensor conversion, TTA, batch inference
- **`model_cache.py`** - Model registry, download with SHA256 verification, cache management
- **`frame_processor.py`** - OpenCV frame extraction/conversion utilities
- **`wrappers/upscale.py`** - Real-ESRGAN (RRDBNet architecture) for AI upscaling
- **`wrappers/interpolate.py`** - RIFE (IFNet + EMD) for AI frame interpolation

Stages with AI implementations (upscale, interpolate, denoise_video) automatically fall back to FFmpeg when PyTorch or models are unavailable.

### GUI
- **PySide6**: Qt for Python (cross-platform GUI)

### CLI
- **Click**: Command-line interface framework
- **Rich**: Rich text formatting

## Future Enhancements

- [x] Real-ESRGAN integration for AI upscaling
- [x] RIFE model for frame interpolation
- [x] AI denoising via Real-ESRGAN (denoise mode)
- [x] AI model cache and download system
- [x] SSIM/PSNR quality metrics (via FFmpeg)
- [ ] Ollama/OpenAI integration for VLM analysis
- [ ] Web-based UI (optional)
- [ ] Plugin system for community stages
- [ ] Batch processing with job scheduling
- [ ] Hardware encoding optimization (NVENC, QSV, VAAPI)
