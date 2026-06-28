# Auto Video Fixer - User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Command-Line Interface](#command-line-interface)
4. [Graphical Interface](#graphical-interface)
5. [Processing Presets](#processing-presets)
6. [Configuration](#configuration)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Installation

### Prerequisites

Before installing Auto Video Fixer, ensure you have:

- **Python 3.10 or higher**
- **FFmpeg** (must be installed and in your PATH)
- **Git** (for cloning the repository)

### Installing FFmpeg

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

#### Windows
1. Download from [FFmpeg Official Site](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH

Verify installation:
```bash
ffmpeg -version
```

### Installing Auto Video Fixer

#### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/auto-video-fixer.git
cd auto-video-fixer

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install with all dependencies
uv pip install -e ".[all]"
```

#### Option 2: Install from PyPI (Stable Release)

```bash
# Install FFmpeg first (see above)

# Install Auto Video Fixer
pip install auto-video-fixer
```

### Verifying Installation

```bash
# Check version
avf --version

# Test CLI
avf --help

# Check GPU support (optional)
avf gpu-info
```

---

## Quick Start

### Process a Single Video

```bash
avf process video.mp4 -p 4k60
```

This will:
1. Analyze the video
2. Stabilize if needed
3. Remove compression artifacts
4. Reduce noise
5. Upscale to 4K resolution
6. Interpolate to 60fps
7. Normalize audio volume
8. Encode with high quality

Output will be saved as `video_enhanced.mp4` in the same directory.

### Process Multiple Videos

```bash
# From a directory
avf process ./videos/ -r -p max_quality -o ./output/

# Multiple specific files
avf process video1.mp4 video2.mkv video3.avi -p 1080p60
```

### Analyze a Video

```bash
avf analyze video.mp4
```

This shows:
- Video properties (resolution, framerate, duration)
- Scene detection
- Content analysis (with VLM if configured)
- Similarity matching

---

## Command-Line Interface

### Available Commands

```
avf [OPTIONS] COMMAND [ARGS]...

Commands:
  process           Process video files with specified settings
  analyze           Analyze a video file
  find-duplicates   Find similar/duplicate videos
  presets           List available presets
  gpu-info          Show GPU information
```

### Process Command

```bash
avf process [PATHS...] [OPTIONS]
```

**Options:**
- `-p, --preset TEXT`: Processing preset name (see [Presets](#processing-presets))
- `-o, --output TEXT`: Output directory
- `-r, --recursive`: Scan directories recursively
- `--dry-run`: Show what would be processed without processing
- `--stage TEXT`: Specific stage to run (can repeat)
- `--threads INT`: Number of concurrent processing threads

**Examples:**

```bash
# Process with preset
avf process video.mp4 -p 4k60

# Process with custom output
avf process video.mp4 -p max_quality -o ./enhanced/

# Dry run to see what would happen
avf process video.mp4 -p 4k60 --dry-run

# Run specific stages only
avf process video.mp4 --stage detect --stage encode

# Process with 4 threads
avf process video.mp4 -p 4k60 --threads 4
```

### Analyze Command

```bash
avf analyze [FILEPATH] [OPTIONS]
```

**Options:**
- `--vlm`: Run VLM content analysis (requires configuration)

**Example:**
```bash
avf analyze video.mp4 --vlm
```

### Find Duplicates Command

```bash
avf find-duplicates REFERENCE DIRECTORY [OPTIONS]
```

**Options:**
- `--threshold FLOAT`: Similarity threshold (0-1, default: 0.95)

**Example:**
```bash
avf find-duplicates original.mp4 ./my_videos/ --threshold 0.9
```

---

## Graphical Interface

### Launching the GUI

```bash
avf-gui
```

Or:
```bash
python -m autovideofixer.gui.main_window
```

### Main Window Features

#### File Selection
- **Add Files...**: Select one or more video files
- **Add Directory...**: Scan a directory for videos
- **Clear Queue**: Remove all files from queue

#### Preset Selection
Choose from built-in presets or create custom ones in settings.

#### Job Queue
- View all queued videos
- See processing status
- Monitor progress
- View output locations

#### Processing Controls
- **Start Processing**: Begin processing all queued jobs
- **Cancel**: Stop processing

### Settings Dialog

Access via **File → Settings...**

#### General Tab
- Output directory
- Max concurrent jobs

#### GPU Tab
- Preferred GPU device (auto, cuda, metal, cpu)

#### Encoding Tab
- Video codec (libx264, libx265, libvpx-vp9, copy)
- CRF quality (0-51, lower = better quality)

---

## Processing Presets

### Built-in Presets

#### max_quality
**Description**: Maximum quality output (slowest)

Use when:
- Archival quality is critical
- You have time to spare
- Output will be displayed on large screens

Settings:
- Target: 4K @ 60fps
- AI upscaling and interpolation
- Aggressive denoising
- CRF 12 (near lossless)

#### 4k60
**Description**: Upscale to 4K at 60fps

Use when:
- Display supports 4K 60Hz
- You want smooth motion
- Source is 1080p or lower

Settings:
- Target: 4K @ 60fps
- Standard enhancement stages
- CRF 18 (high quality)

#### 4k30
**Description**: Upscale to 4K at 30fps

Use when:
- Display supports 4K but not 60Hz
- You want sharp image
- Source is 1080p or lower

Settings:
- Target: 4K @ 30fps
- Standard enhancement stages
- CRF 18 (high quality)

#### 1080p60
**Description**: Smooth 1080p60 output

Use when:
- Display is 1080p 60Hz
- You want smooth motion
- Source is lower resolution or framerate

Settings:
- Target: 1080p @ 60fps
- Frame interpolation
- CRF 20 (high quality)

#### size_reduction
**Description**: Reduce file size with acceptable quality loss

Use when:
- Storage is limited
- You need smaller files
- Quality is secondary to size

Settings:
- CRF 28 (smaller files)
- No upscaling or interpolation
- Minimal enhancement
- Target: <10% quality loss

#### remux_only
**Description**: Change container format without re-encoding

Use when:
- You just need to change format
- You want to preserve original quality
- Speed is important

Settings:
- Copy all streams
- No re-encoding
- Instant processing

#### hdr_enhance
**Description**: Convert and enhance HDR content

Use when:
- Source is HDR (HDR10, Dolby Vision)
- You want to convert to SDR
- You need HDR optimization

Settings:
- HDR to SDR conversion
- Enhanced encoding
- CRF 22

### Creating Custom Presets

Custom presets are stored in:
- **Linux**: `~/.config/auto-video-fixer/presets/`
- **macOS**: `~/Library/Application Support/auto-video-fixer/presets/`
- **Windows**: `%APPDATA%\auto-video-fixer\presets\`

Format (JSON):
```json
{
  "name": "my_preset",
  "display_name": "My Custom Preset",
  "description": "Custom processing preset",
  "target_resolution": [1920, 1080],
  "target_framerate": 30.0,
  "video_codec": "libx264",
  "audio_codec": "aac",
  "crf": 20,
  "preset": "medium",
  "enable_stages": {
    "detect": true,
    "upscale": true,
    "encode": true
  }
}
```

---

## Configuration

### Configuration File Location

- **Linux**: `~/.config/auto-video-fixer/config.yaml`
- **macOS**: `~/Library/Application Support/auto-video-fixer/config.yaml`
- **Windows**: `%APPDATA%\auto-video-fixer\config.yaml`

### Configuration Options

#### General Settings
```yaml
general:
  output_dir: null  # null = same as input
  temp_dir: null    # null = system temp
  max_concurrent_jobs: 1
  log_level: INFO
  overwrite: false
```

#### GPU Settings
```yaml
gpu:
  auto_detect: true
  preferred_device: auto  # auto, cuda, metal, cpu
  memory_limit_gb: null
```

#### FFmpeg Settings
```yaml
ffmpeg:
  binary: null  # null = auto-detect
  hwaccel: auto  # auto, cuda, vaapi, qsv, none
  threads: 0  # 0 = auto
```

#### Quality Settings
```yaml
quality:
  vmaf_model: vmaf_v0.6.1
  vmaf_features: psnr,ssim,ms_ssim,fast
  quality_target:
    mode: none  # none, min, avg, max, target
    target: 95.0
    max_loss_pct: 5.0
```

#### Stage Settings
```yaml
stages:
  upscale:
    enabled: true
    ai_model: RealESRGAN_x4plus
    traditional_method: superres
  interpolate:
    enabled: true
    ai_model: rife_v4.6
    traditional_method: minterpolate
  # ... other stages
```

### Editing Configuration

#### Option 1: Edit YAML File Directly

Open the config file in a text editor and modify values.

#### Option 2: Use GUI Settings

1. Launch GUI
2. Go to **File → Settings**
3. Modify settings
4. Click OK (automatically saves)

#### Option 3: Command-Line Override

Some settings can be overridden via command line:
```bash
avf process video.mp4 --threads 4
```

---

## Advanced Usage

### Custom Processing Stages

Run specific stages only:
```bash
avf process video.mp4 --stage detect --stage upscale --stage encode
```

### Quality Targeted Processing

Set quality targets in configuration:
```yaml
quality:
  quality_target:
    mode: target
    target: 95.0  # VMAF score
    target_resolution: [3840, 2160]
    target_framerate: 60.0
```

### Hardware Acceleration

Enable GPU encoding:
```yaml
ffmpeg:
  hwaccel: cuda  # or vaapi, qsv, etc.
```

Check available hardware:
```bash
avf gpu-info
```

### Directory Monitoring

Process all videos in a directory:
```bash
avf process ./videos/ -r -p max_quality -o ./output/
```

### Batch Processing with Priority

Process files with different priorities:
```bash
# High priority files first
avf process important.mp4 --priority 10
avf process normal.mp4 --priority 0
```

### Logging and Debugging

Enable debug logging:
```bash
# Set in config
log_level: DEBUG
```

Or use environment variable:
```bash
export AVF_LOG_LEVEL=DEBUG
avf process video.mp4
```

### API and VLM Integration

Configure VLM for content analysis:
```yaml
analysis:
  vlm:
    enabled: true
    provider: ollama  # or openai, api
    model: llava
    api_url: http://localhost:11434
    api_key: ""
```

---

## Troubleshooting

### Common Issues

#### "ffmpeg not found"

**Problem**: FFmpeg is not installed or not in PATH.

**Solution**:
1. Install FFmpeg (see [Installing FFmpeg](#installing-ffmpeg))
2. Verify installation: `ffmpeg -version`
3. Add to PATH if needed

#### "CUDA not available"

**Problem**: NVIDIA GPU not detected or drivers not installed.

**Solution**:
1. Check GPU: `nvidia-smi`
2. Install latest NVIDIA drivers
3. Verify CUDA: `nvcc --version`
4. Set `gpu.preferred_device: cpu` in config to disable

#### "Out of memory"

**Problem**: Processing large videos with AI models.

**Solution**:
1. Reduce batch size
2. Use CPU instead of GPU
3. Process smaller segments
4. Free up system memory

#### "Encoding failed"

**Problem**: FFmpeg encoding error.

**Solution**:
1. Check FFmpeg version: `ffmpeg -version`
2. Try different codec: `libx264` or `libx265`
3. Lower quality: increase CRF value
4. Check disk space

#### "Stage failed"

**Problem**: A processing stage encountered an error.

**Solution**:
1. Check log output for details
2. Try with `--dry-run` first
3. Disable problematic stage in config
4. Update to latest version

### Getting Help

1. **Check Logs**: Enable debug logging for detailed output
2. **GitHub Issues**: [Report bugs](https://github.com/yourusername/auto-video-fixer/issues)
3. **Discord**: Join community server
4. **Documentation**: Check [docs/](../docs/) folder

### Performance Tips

1. **Use GPU**: Enable CUDA/Metal for AI processing
2. **Limit Threads**: Don't set too high (start with 2-4)
3. **SSD Storage**: Use SSD for temp files
4. **Close Other Apps**: Free up RAM and GPU memory
5. **Process in Batches**: Don't overload the system

---

## FAQ

### Q: What video formats are supported?

**A**: Auto Video Fixer supports all formats FFmpeg supports, including:
- MP4, MKV, AVI, MOV, WMV, FLV, WebM
- MPEG, MPG, 3GP, OGV, TS, VOB
- And many more

### Q: How long does processing take?

**A**: Depends on:
- Video length and resolution
- Preset complexity
- Hardware (CPU/GPU)
- AI models enabled

Rough estimates (1080p video):
- **Basic encoding**: 1-2x realtime
- **With stabilization**: 2-3x realtime
- **With AI upscaling**: 10-20x realtime (GPU)

### Q: Can I use my own AI models?

**A**: Yes! Create custom stages following the [developer documentation](./DEVELOPER.md).

### Q: Is there a web interface?

**A**: Currently only GUI and CLI. Web interface planned for future releases.

### Q: How do I create custom presets?

**A**: Create a JSON file in the presets directory (see [Creating Custom Presets](#creating-custom-presets)).

### Q: Can I process videos without internet?

**A**: Yes! All processing is local. VLM integration requires internet for cloud APIs.

### Q: What's the difference between AI and traditional processing?

**A**:
- **AI**: Better quality, slower, requires GPU
- **Traditional**: Faster, good quality, works on CPU

### Q: How do I update Auto Video Fixer?

**A**:
```bash
# From source
git pull
uv pip install -e ".[all]"

# From PyPI
pip install --upgrade auto-video-fixer
```

---

## Additional Resources

- **GitHub Repository**: [https://github.com/yourusername/auto-video-fixer](https://github.com/yourusername/auto-video-fixer)
- **Issue Tracker**: [https://github.com/yourusername/auto-video-fixer/issues](https://github.com/yourusername/auto-video-fixer/issues)
- **Discussions**: [https://github.com/yourusername/auto-video-fixer/discussions](https://github.com/yourusername/auto-video-fixer/discussions)
- **Roadmap**: [ROADMAP.md](./ROADMAP.md)
- **Developer Guide**: [DEVELOPER.md](./DEVELOPER.md)

---

*Last updated: June 2026*
