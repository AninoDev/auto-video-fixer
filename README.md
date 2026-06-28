# Auto Video Fixer

Automated video enhancement and processing suite with AI-powered upscaling, frame interpolation, and intelligent pipeline optimization.

## Features

- **AI-Powered Enhancement**: Upscaling (Real-ESRGAN), frame interpolation (RIFE), denoising
- **Traditional Processing**: FFmpeg-based deblocking, stabilization, normalization
- **Intelligent Pipeline**: Automatic stage ordering for optimal quality and speed
- **VMAF Quality Control**: Adjustable quality targets with perceptual quality metrics
- **Multi-Format Support**: Wide range of video/audio codecs and containers
- **Hardware Acceleration**: GPU support for encoding and AI inference
- **Cross-Platform**: Windows, Linux, macOS
- **GUI + CLI**: Full-featured Qt GUI and command-line interface
- **Modular Architecture**: Easy to add new processing stages
- **Preset System**: Predefined profiles for common use cases

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd auto-video-fixer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[all]"
```

### Prerequisites

- **FFmpeg**: Must be installed and in PATH (https://ffmpeg.org/)
- **Python**: 3.10 or higher
- **GPU** (optional): NVIDIA CUDA, Apple Metal, or Intel QuickSync for hardware acceleration

## Usage

### GUI

```bash
avf-gui
```

Or:
```bash
python -m autovideofixer.gui.main_window
```

### CLI

```bash
# Process files with a preset
avf process video.mp4 -p 4k60

# Process multiple files from a directory
avf process ./videos/ -r -p max_quality -o ./output/

# Dry run to see what would be processed
avf process video.mp4 --dry-run

# Analyze a video
avf analyze video.mp4

# Find similar/duplicate videos
avf find-duplicates reference.mp4 ./library/

# List available presets
avf list-presets

# Show GPU info
avf gpu-info
```

## Presets

| Preset | Description |
|--------|-------------|
| `max_quality` | Maximum quality output (slowest) |
| `4k60` | Upscale to 4K at 60fps |
| `4k30` | Upscale to 4K at 30fps |
| `1080p60` | Smooth 1080p60 output |
| `size_reduction` | Reduce file size with acceptable quality loss |
| `remux_only` | Just change container format |
| `hdr_enhance` | Convert and enhance HDR content |

## Architecture

```
src/autovideofixer/
├── __init__.py              # Package root
├── config.py                # Configuration system
├── logger.py                # Logging utilities
├── core/
│   ├── __init__.py
│   ├── pipeline.py          # Pipeline orchestrator
│   ├── quality.py           # VMAF quality estimation
│   ├── analysis.py          # Video analysis, VLM, duplicate detection
│   ├── ffmpeg_utils.py      # FFmpeg integration
│   ├── presets.py           # Processing presets
│   └── stages/
│       ├── __init__.py
│       ├── base.py          # Base stage class + registry
│       ├── detect.py        # Analysis/detection
│       ├── stabilize.py     # Video stabilization
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
│   ├── __init__.py
│   └── main_window.py       # Qt GUI application
└── cli/
    ├── __init__.py
    └── cli.py               # Click-based CLI
```

## Adding a New Processing Stage

1. Create a new file in `src/autovideofixer/core/stages/`
2. Subclass `BaseStage`
3. Implement `execute()` method
4. Register with the `@register_stage` decorator
5. Add to the default pipeline order in `Config.DEFAULTS`

Example:

```python
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus, register_stage

@register_stage
class MyNewStage(BaseStage):
    name = "my_stage"
    display_name = "My New Stage"
    description = "Does something cool"
    category = "enhancement"
    priority = 25
    supports_gpu = True

    def should_run(self, input_info):
        if not self.is_enabled():
            return False, "Disabled"
        return True, None

    def execute(self, input_path, output_path, progress_callback, **kwargs):
        import time
        start = time.time()
        self._report_progress(0.0, "Processing...", progress_callback)
        # ... do processing ...
        self._report_progress(1.0, "Done", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            duration_sec=time.time() - start,
        )
```

## Configuration

Configuration is stored at:
- **Linux**: `~/.config/auto-video-fixer/config.yaml`
- **macOS**: `~/Library/Application Support/auto-video-fixer/config.yaml`
- **Windows**: `%APPDATA%\auto-video-fixer\config.yaml`

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/
```

## License

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).

See the [LICENSE](LICENSE) file for details.

### Key Points of AGPL

- You may copy, distribute and modify the software as long as you track all changes
- You may distribute copies of the software (with or without modifications)
- If you modify the software, you must release the modified source code under the same license
- If you run the software on a server, you must make the complete source code available to users of that server
- You may not impose any further restrictions on the exercise of the rights granted herein
