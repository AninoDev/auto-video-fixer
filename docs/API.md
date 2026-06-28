# Auto Video Fixer - API Documentation

## Overview

Auto Video Fixer provides a modular API for video processing. This document describes the public API for developers.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Pipeline API](#pipeline-api)
- [Stage API](#stage-api)
- [Quality API](#quality-api)
- [Analysis API](#analysis-api)
- [Preset API](#preset-api)
- [Examples](#examples)

---

## Installation

```bash
pip install auto-video-fixer
```

Or from source:

```bash
git clone https://github.com/yourusername/auto-video-fixer.git
cd auto-video-fixer
pip install -e ".[all]"
```

---

## Quick Start

```python
from autovideofixer.config import Config
from autovideofixer.core.pipeline import Pipeline

# Create configuration
config = Config()

# Create pipeline
pipeline = Pipeline(config)

# Add a job
job = pipeline.add_job("input.mp4")

# Process
result = pipeline.execute_job(job)

print(f"Success: {result.success}")
print(f"Output: {result.output_path}")
```

---

## Configuration

### Config Class

```python
from autovideofixer.config import Config

# Create config with defaults
config = Config()

# Create config with custom path
config = Config("/path/to/config.yaml")

# Get values
max_jobs = config.get("general", "max_concurrent_jobs")
vmaf_model = config.get("quality", "vmaf_model")

# Set values
config.set(2, "general", "max_concurrent_jobs")
config.set(95.0, "quality", "quality_target", "target")

# Save to disk
config.save()
```

### Configuration Structure

```python
{
    "general": {
        "output_dir": str | None,
        "temp_dir": str | None,
        "max_concurrent_jobs": int,
        "log_level": str,
        "overwrite": bool,
    },
    "gpu": {
        "auto_detect": bool,
        "preferred_device": str,
        "memory_limit_gb": float | None,
    },
    "ffmpeg": {
        "binary": str | None,
        "hwaccel": str,
        "threads": int,
    },
    "quality": {
        "vmaf_model": str,
        "vmaf_features": str,
        "quality_target": {
            "mode": str,
            "target": float,
            "max_loss_pct": float,
        },
    },
    "stages": {
        "stage_name": {
            "enabled": bool,
            # stage-specific settings
        },
    },
}
```

---

## Pipeline API

### Pipeline Class

The main orchestrator for video processing.

```python
from autovideofixer.core.pipeline import Pipeline, Job, JobResult

pipeline = Pipeline(config)
```

#### Methods

##### `add_job(input_path, output_path=None, stage_names=None, overrides=None, priority=0) -> Job`

Add a processing job to the queue.

**Parameters:**
- `input_path` (str): Path to input video file
- `output_path` (str | None): Desired output path (default: input_enhanced.ext)
- `stage_names` (list[str] | None): Specific stages to run (default: auto-determine)
- `overrides` (dict[str, dict] | None): Per-stage configuration overrides
- `priority` (int): Job priority (higher = processed first)

**Returns:** `Job` instance

**Example:**
```python
job = pipeline.add_job(
    "input.mp4",
    "output.mp4",
    stage_names=["detect", "upscale", "encode"],
    overrides={
        "upscale": {"method": "ai", "scale_factor": 2.0}
    },
    priority=10
)
```

##### `add_files(paths: list[str]) -> list[Job]`

Add multiple files or directories to the queue.

**Parameters:**
- `paths` (list[str]): List of file or directory paths

**Returns:** List of `Job` instances

**Example:**
```python
jobs = pipeline.add_files([
    "video1.mp4",
    "video2.mkv",
    "/path/to/videos/"
])
```

##### `execute_job(job: Job) -> JobResult`

Execute a single job.

**Parameters:**
- `job` (Job): Job to execute

**Returns:** `JobResult` with outcome

**Example:**
```python
result = pipeline.execute_job(job)
if result.success:
    print(f"Output: {result.output_path}")
```

##### `execute_all(callback=None) -> list[JobResult]`

Execute all jobs in the queue.

**Parameters:**
- `callback` (callable | None): Function(job, result) called for each completed job

**Returns:** List of `JobResult` instances

**Example:**
```python
def on_complete(job, result):
    print(f"{job.input_path}: {'OK' if result.success else 'FAIL'}")

results = pipeline.execute_all(callback=on_complete)
```

##### `cancel()`

Cancel all running jobs.

##### `clear_queue()`

Remove all jobs from the queue.

### Job Class

Represents a processing job.

```python
from autovideofixer.core.pipeline import Job

job = Job(
    input_path="input.mp4",
    output_path="output.mp4",
    stages=["detect", "encode"],
    priority=0,
)
```

**Attributes:**
- `input_path` (str): Input file path
- `output_path` (str | None): Output file path
- `stages` (list[str]): Stages to execute
- `stage_overrides` (dict[str, dict]): Per-stage overrides
- `priority` (int): Job priority
- `status` (PipelineStatus): Current status
- `progress` (float): Progress (0.0-1.0)
- `current_stage` (str | None): Currently running stage
- `result` (JobResult | None): Final result

### JobResult Class

Result of a completed job.

```python
from autovideofixer.core.pipeline import JobResult

result = JobResult(
    input_path="input.mp4",
    output_path="output.mp4",
    stage_results={"detect": StageResult(...), "encode": StageResult(...)},
    success=True,
)
```

**Attributes:**
- `input_path` (str): Input file path
- `output_path` (str | None): Output file path
- `stage_results` (dict[str, StageResult]): Results per stage
- `total_duration` (float): Total processing time
- `errors` (list[str]): Error messages
- `skipped` (list[str]): Skipped stages
- `input_info` (dict): Input video metadata
- `output_info` (dict): Output video metadata
- `success` (bool): Whether job succeeded

**Methods:**
- `all_stages_passed` (bool): All stages succeeded
- `output_size` (int): Output file size
- `input_size` (int): Input file size
- `size_ratio` (float): Output/Input size ratio

---

## Stage API

### BaseStage Class

Abstract base class for all processing stages.

```python
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus

class MyStage(BaseStage):
    name = "my_stage"
    display_name = "My Stage"
    description = "Does something"
    category = "enhancement"
    priority = 25
    supports_gpu = True
    
    def should_run(self, input_info) -> tuple[bool, str | None]:
        # Determine if stage should run
        return True, None
    
    def execute(self, input_path, output_path, progress_callback=None, **kwargs) -> StageResult:
        # Execute processing
        return StageResult(status=StageStatus.COMPLETED, output_path=output_path)
```

**Attributes:**
- `name` (str): Unique stage identifier
- `display_name` (str): Human-readable name
- `description` (str): Stage description
- `requires_input` (bool): Requires input file
- `produces_output` (bool): Produces output file
- `supports_gpu` (bool): Can use GPU acceleration
- `supports_hardware_encoding` (bool): Can use HW encoding
- `priority` (int): Execution order (lower = earlier)
- `category` (str): Stage category
- `can_parallelize` (bool): Can run in parallel
- `needs_intermediate` (bool): Must run on intermediate file

**Methods:**
- `is_enabled()` -> bool: Check if stage is enabled
- `should_run(input_info)` -> tuple[bool, str | None]: Determine if stage should run
- `execute(input_path, output_path, progress_callback, **kwargs)` -> StageResult: Execute stage
- `estimate_complexity(input_info)` -> float: Estimate processing complexity
- `cleanup(path)` -> None: Clean up temporary files

### StageResult Dataclass

Result of a stage execution.

```python
from autovideofixer.core.stages.base import StageResult, StageStatus

result = StageResult(
    status=StageStatus.COMPLETED,
    output_path="output.mp4",
    metadata={"key": "value"},
    duration_sec=10.5,
)
```

**Attributes:**
- `status` (StageStatus): Stage status
- `output_path` (str | None): Output file path
- `metadata` (dict): Additional metadata
- `error` (str | None): Error message (if failed)
- `duration_sec` (float): Processing duration
- `skipped_reason` (str | None): Skip reason (if skipped)

**Properties:**
- `success` (bool): True if completed or skipped

### StageStatus Enum

```python
from autovideofixer.core.stages.base import StageStatus

StageStatus.PENDING      # Not yet started
StageStatus.RUNNING      # Currently executing
StageStatus.COMPLETED    # Successfully finished
StageStatus.SKIPPED      # Skipped (not applicable)
StageStatus.FAILED       # Execution failed
StageStatus.CANCELLED    # Cancelled by user
```

### Stage Registry

```python
from autovideofixer.core.stages.base import register_stage, create_stage, list_stages

# Register a stage
@register_stage
class MyStage(BaseStage):
    # ...

# Get all stages
stages = list_stages()

# Create a stage instance
stage = create_stage("my_stage", config)
```

---

## Quality API

### Quality Estimation

```python
from autovideofixer.core.quality import estimate_quality_vmaf, QualityResult

result = estimate_quality_vmaf(
    reference="original.mp4",
    distorted="processed.mp4",
)

print(f"VMAF: {result.vmaf_score}")
print(f"PSNR: {result.psnr}")
print(f"SSIM: {result.ssimm}")
```

### QualityResult

```python
from autovideofixer.core.quality import QualityResult, QualityMode

result = QualityResult(
    vmaf_score=95.0,
    psnr=40.0,
    ssimm=0.95,
    mode=QualityMode.TARGET,
    target=90.0,
)

# Get score based on mode
score = result.score

# Check if meets target
if result.meets_target():
    print("Quality acceptable")
```

**QualityMode Enum:**
- `NONE`: No quality target
- `MIN`: Use minimum score
- `AVG`: Use average score
- `MAX`: Use maximum score
- `TARGET`: Use against target value

### Quality Loss Estimation

```python
from autovideofixer.core.quality import estimate_quality_loss

acceptable, reason = estimate_quality_loss(
    original_size=1000000,
    new_size=800000,
    quality=result,
    max_loss_pct=10.0,
)

if acceptable:
    print(f"Acceptable: {reason}")
```

---

## Analysis API

### Video Analysis

```python
from autovideofixer.core.analysis import VideoAnalyzer, is_video_file

# Check if file is video
if is_video_file("video.mp4"):
    print("Valid video file")

# Analyze video
analyzer = VideoAnalyzer(config)
analysis = analyzer.analyze("video.mp4")

print(f"Resolution: {analysis.resolution}")
print(f"Duration: {analysis.duration}")
print(f"Scenes: {analysis.total_scenes}")
```

### VideoAnalysis

```python
from autovideofixer.core.analysis import VideoAnalysis

analysis = VideoAnalysis(
    filepath="video.mp4",
    filename="video.mp4",
    duration=120.5,
    resolution=(1920, 1080),
    framerate=30.0,
    has_video=True,
    has_audio=True,
    is_hdr=False,
    total_scenes=5,
    scenes=[...],
    vlm_summary="A video of...",
    vlm_tags=["nature", "outdoor"],
)
```

### Scene Detection

```python
scenes = analyzer.detect_events("video.mp4")

for scene in scenes:
    print(f"Scene: {scene.start_time:.1f}s - {scene.end_time:.1f}s")
```

### Duplicate Detection

```python
similar = analyzer.find_similar(
    reference="original.mp4",
    candidates=["copy1.mp4", "copy2.mp4"],
    threshold=0.95,
)

for path, similarity in similar:
    print(f"{path}: {similarity*100:.1f}% similar")
```

---

## Preset API

### Preset Management

```python
from autovideofixer.core.presets import get_preset, list_presets, save_preset

# List all presets
presets = list_presets()

# Get specific preset
preset = get_preset("4k60")

# Convert to config
config_dict = preset.to_config()
```

### Preset Class

```python
from autovideofixer.core.presets import Preset

preset = Preset(
    name="my_preset",
    display_name="My Preset",
    description="Custom preset",
    target_resolution=(3840, 2160),
    target_framerate=60.0,
    video_codec="libx264",
    audio_codec="aac",
    crf=18,
    enable_stages={
        "detect": True,
        "upscale": True,
        "encode": True,
    },
)

# Save to disk
save_preset(preset, "/path/to/preset.json")
```

---

## FFmpeg Utilities

### Video Probing

```python
from autovideofixer.core.ffmpeg_utils import probe, get_video_info

# Get detailed info
info = probe("video.mp4")
print(f"Resolution: {info.resolution}")
print(f"Framerate: {info.framerate}")
print(f"Duration: {info.duration}")

# Get simplified info
info_dict = get_video_info("video.mp4")
```

### Hardware Acceleration

```python
from autovideofixer.core.ffmpeg_utils import detect_hardware_acceleration, resolve_hwaccel

# Detect available accelerations
hwaccels = detect_hardware_acceleration()
print(f"Available: {hwaccels}")

# Resolve preferred acceleration
accel = resolve_hwaccel("auto")
print(f"Using: {accel}")
```

### Running FFmpeg

```python
from autovideofixer.core.ffmpeg_utils import run_ffmpeg

result = run_ffmpeg([
    "-i", "input.mp4",
    "-c:v", "libx264",
    "-c:a", "aac",
    "output.mp4"
])

if result.returncode == 0:
    print("Success")
else:
    print(f"Failed: {result.stderr}")
```

---

## Examples

### Basic Processing

```python
from autovideofixer.config import Config
from autovideofixer.core.pipeline import Pipeline

config = Config()
pipeline = Pipeline(config)

job = pipeline.add_job("input.mp4")
result = pipeline.execute_job(job)

print(f"Output: {result.output_path}")
```

### Custom Processing

```python
from autovideofixer.config import Config
from autovideofixer.core.pipeline import Pipeline

config = Config()
config.set(2, "general", "max_concurrent_jobs")

pipeline = Pipeline(config)

job = pipeline.add_job(
    "input.mp4",
    "output.mp4",
    stage_names=["detect", "upscale", "encode"],
    overrides={
        "upscale": {"method": "ai", "scale_factor": 2.0}
    }
)

result = pipeline.execute_job(job)
```

### Batch Processing

```python
from autovideofixer.config import Config
from autovideofixer.core.pipeline import Pipeline

config = Config()
pipeline = Pipeline(config)

# Add multiple files
jobs = pipeline.add_files(["video1.mp4", "video2.mkv", "./videos/"])

# Process with callback
def on_complete(job, result):
    status = "OK" if result.success else "FAIL"
    print(f"{job.input_path}: {status}")

results = pipeline.execute_all(callback=on_complete)
```

### Quality Control

```python
from autovideofixer.config import Config
from autovideofixer.core.pipeline import Pipeline
from autovideofixer.core.quality import estimate_quality_vmaf

config = Config()
config.set(95.0, "quality", "quality_target", "target")

pipeline = Pipeline(config)
job = pipeline.add_job("input.mp4")
result = pipeline.execute_job(job)

# Verify quality
if result.output_path:
    quality = estimate_quality_vmaf("input.mp4", result.output_path)
    print(f"VMAF: {quality.vmaf_score}")
```

### Custom Stage

```python
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus, register_stage

@register_stage
class MyFilterStage(BaseStage):
    name = "my_filter"
    display_name = "My Filter"
    description = "Applies custom filter"
    category = "enhancement"
    priority = 30
    
    def should_run(self, input_info):
        return True, None
    
    def execute(self, input_path, output_path, progress_callback=None, **kwargs):
        import time
        start = time.time()
        
        self._report_progress(0.0, "Processing...", progress_callback)
        
        # Your processing logic
        # ...
        
        self._report_progress(1.0, "Done", progress_callback)
        
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            duration_sec=time.time() - start,
        )
```

---

## Error Handling

### Stage Failures

```python
result = pipeline.execute_job(job)

if not result.success:
    for stage_name, stage_result in result.stage_results.items():
        if stage_result.status == StageStatus.FAILED:
            print(f"{stage_name} failed: {stage_result.error}")
```

### Pipeline Errors

```python
try:
    result = pipeline.execute_job(job)
except Exception as e:
    print(f"Pipeline error: {e}")
```

### Validation

```python
from autovideofixer.core.ffmpeg_utils import probe

try:
    info = probe("video.mp4")
except FileNotFoundError:
    print("Video file not found")
except Exception as e:
    print(f"Probe failed: {e}")
```

---

## Performance Tips

1. **Use GPU**: Enable CUDA/Metal for AI processing
2. **Limit Threads**: Start with 2-4 threads
3. **Skip Unnecessary Stages**: Let `should_run()` optimize
4. **Batch Processing**: Process multiple files concurrently
5. **Monitor Progress**: Use progress callbacks

---

## Best Practices

1. **Always check return values**: Stages can fail
2. **Use progress callbacks**: For responsive UIs
3. **Handle errors gracefully**: Don't crash on failures
4. **Clean up temp files**: Pipeline handles this automatically
5. **Test with small files first**: Validate before processing large files

---

## Support

- **Documentation**: [docs/](./)
- **Issues**: [GitHub Issues](https://github.com/yourusername/auto-video-fixer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/auto-video-fixer/discussions)

---

*Last updated: June 2026*
