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

- **Python 3.14 or higher**
- **FFmpeg** (must be installed and in your PATH)
- **Git** (for cloning the repository)
- **PyTorch** (optional, for AI upscaling/interpolation) - install with `uv pip install "auto-video-fixer[ai]"`

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

# With AI support (optional, requires PyTorch):
pip install "auto-video-fixer[ai]"
```

### Verifying Installation

```bash
# Check version
avf --version

# Test CLI
avf --help

# Check GPU support (optional)
avf gpu-info

# Check AI models (requires PyTorch)
avf model-info
```

### Installing AI Models

AI models are downloaded automatically on first use. To pre-download models:

```bash
# Download models manually
avf model-download --model real-esrgan-x4plus
avf model-download --model rife-v4.6
```

Models are stored in the data directory:
- **Linux**: `~/.local/share/auto-video-fixer/models/`
- **macOS**: `~/Library/Application Support/auto-video-fixer/models/`
- **Windows**: `%APPDATA%\auto-video-fixer\models\`

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
  model-info        Show AI model information and download status
  model-download    Download an AI model
```

### Model Management Commands

```bash
# Show available AI models and their status
avf model-info

# Download a specific model
avf model-download --model real-esrgan-x4plus

# Download with custom URL
avf model-download --model my-custom-model --url https://example.com/model.pth

# Force re-download (overwrite existing)
avf model-download --model rife-v4.6 --force
```

Available models:
- `real-esrgan-x4plus` - General upscaling (4x)
- `real-esrgan-x2plus` - General upscaling (2x)
- `real-esrgan-x4plus-anime-6b` - Anime/cartoon upscaling
- `rife-v4.6` - Frame interpolation (default)
- `rife-v4.11` - Frame interpolation (higher quality, slower)

### Process Command

```bash
avf process [PATHS...] [OPTIONS]
```

**Options:**
- `-p, --preset TEXT`: Processing preset name (see [Presets](#processing-presets))
- `-o, --output TEXT`: Output directory
- `-r, --recursive`: Scan directories recursively
- `--symlinks [files_only|follow|skip]`: How directory scanning treats symlinks (default
  `files_only`). See [Symlinks When Scanning Directories](#symlinks-when-scanning-directories).
- `--dry-run`: Show what would be processed without processing
- `--stage TEXT`: Specific stage to run (can repeat)
- `--threads INT`: Number of concurrent processing threads
- `--color / --no-color`: Force console colour on/off (see [Console Colour](#console-colour))

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

### Live Progress Bars

When run in a real terminal, `process` shows two live progress bars: a batch bar (how many
videos are done, including the current one's partial progress) and a per-file bar (the current
video's own progress). Both are on by default and disappear automatically when output is piped
or redirected. Turn either off explicitly with `--no-progress-batch` / `--no-progress-file` (or
their `--progress-batch` / `--progress-file` counterparts to force them on), or persist the
choice via `reporting.progress_batch` / `reporting.progress_file` in your config file.

```bash
# Suppress both bars, e.g. for a cleaner log when redirecting to a file by hand
avf process video.mp4 -p 4k60 --no-progress-batch --no-progress-file
```

### Console Colour

By default (`auto`) Rich only colours `process`'s console output when stdout/stderr is a real
terminal -- the common case where this bites is `avf process ... | tee run.log`: piping stdout
through `tee` makes it a pipe rather than a TTY, so Rich silently drops to plain text even though
you're still watching a real terminal on the other end of `tee`. Pass `--color` to force colour
back on for that case, or `--no-color` to force it off (e.g. a terminal that mishandles ANSI, or
you just want clean output to eyeball). Persist the choice via `reporting.color` (`auto` |
`always` | `never`) in your config file; the CLI flag wins when both are given.

```bash
# Keep colour even though tee makes stdout a pipe
avf process video.mp4 -p 4k60 --color | tee run.log
```

Colour and the live progress bars above are independent controls: `--color`/`reporting.color:
always` does **not** re-enable the progress bars for piped/redirected output -- those stay gated
on stdout actually being a real terminal, since their redraw control codes would otherwise
corrupt the piped file. Forcing colour on only affects what the console text itself looks like.

### Reading Inputs From a List File

Besides typing paths directly, `process` can read them from one or more `--from-file` list
files, parsed by a pluggable parser selected with `--from-file-parser` (default `shlex`). This
is handy for batches you've already selected elsewhere, or manifests that pair specific inputs
with specific output names.

```bash
# A file with one path per return -- works with either the default (shlex)
# or the "lines" parser (lines also supports "#" comments and blank lines):
cat > batch.txt <<EOF
/videos/clip1.mp4
/videos/clip2.mp4
EOF
avf process --from-file batch.txt -p 4k60

# Drag-and-drop-style paste (e.g. selecting files in Dolphin and dropping them
# onto a Konsole terminal produces a run of quoted, whitespace/newline-
# separated paths) -- this is exactly what the default "shlex" parser handles:
cat > dropped.txt <<'EOF'
"/videos/My Trip.mp4" "/videos/Birthday Party.mp4"
EOF
avf process --from-file dropped.txt -p 1080p60

# A manifest listing per-file comments and blanks (--from-file-parser lines):
cat > manifest.txt <<'EOF'
# Weekend footage
/videos/clip1.mp4

# Skip clip2, it's already processed
/videos/clip3.mp4
EOF
avf process --from-file-parser lines --from-file manifest.txt -p size_reduction

# A CSV manifest giving each input its own output path/name:
cat > jobs.csv <<EOF
input,output
/videos/clip1.mp4,/enhanced/clip1_final.mp4
/videos/clip2.mp4,/enhanced/clip2_final.mp4
EOF
avf process --from-file-parser csv --from-file jobs.csv

# Inline override: pick the parser for just this one file without changing
# the stateful --from-file-parser (useful when combining several list files
# in different formats in one command):
avf process --from-file "csv:jobs.csv" --from-file batch.txt

# Switching parsers mid-command -- everything after each --from-file-parser
# uses that parser until the next one:
avf process \
  --from-file-parser lines --from-file manifest.txt \
  --from-file-parser json  --from-file jobs.json
```

Positional `PATHS` and `--from-file` inputs can be combined in the same command, and directory
entries in a list file expand exactly like a directory typed on the command line does. See
AGENTS.md's "Input file lists & pluggable parsers" section for the full parser reference and how
to add a custom parser.

### Analyze Command

```bash
avf analyze [FILEPATH] [OPTIONS]
```

**Options:**
- `--vlm`: Run VLM content analysis (requires configuration)
- `--min-sample-frames N`, `--max-sample-frames N`, `--sample-interval SECONDS`: how many frames
  are sent to the VLM (see [How many frames are sent](#how-many-frames-are-sent-and-how-big)); the
  result shows frames sent, duplicates dropped, frame size and estimated tokens
- `-r, --recursive`: Scan directories recursively
- `--symlinks [files_only|follow|skip]`: How directory scanning treats symlinks (default
  `files_only`). See [Symlinks When Scanning Directories](#symlinks-when-scanning-directories).

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
- `--symlinks [files_only|follow|skip]`: How directory scanning treats symlinks (default
  `files_only`).

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

#### AI Tab
- AI Upscaler model (Real-ESRGAN x2plus, x4plus, anime 6B)
- AI Interpolator model (RIFE v4.6, v4.11)
- TTA mode for upscaling (0-7, higher = better quality but slower)
- Model cache management (view, clear cached models)

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
    ai_model: RealESRGAN_x4plus  # RealESRGAN_x4plus, RealESRGAN_x2plus, RealESRGAN_x4plus_anime_6B
    traditional_method: superres
    scale_factor: 4
    tta_mode: 0  # Test-time augmentation mode (0=off, 1-7)
  interpolate:
    enabled: true
    ai_model: rife_v4.6  # rife_v4.6, rife_v4.11
    traditional_method: minterpolate
  denoise_video:
    enabled: true
    ai_model: RealESRGAN_x4plus  # Use Real-ESRGAN in denoise mode
    traditional_method: hqdn3d
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

### Deinterlacing

Interlaced source video (combing artefacts visible on motion) can be converted to progressive
frames with the `deinterlace` stage. It's **off by default** -- automatic detection (via ffmpeg's
`idet` filter) is a heuristic that can misclassify detailed progressive motion, and deinterlacing
progressive content is destructive, so this is opt-in only:

```bash
avf process interlaced.mkv --enable-stage deinterlace
```

Or in `config.yaml`:
```yaml
stages:
  deinterlace:
    enabled: true
    mode: field          # frame | field | bob -- field (default) preserves full motion
    filter: bwdif        # bwdif | yadif | w3fdif | estdif
    field_order: auto     # auto | tff | bff -- auto uses idet detection
    min_interlaced_ratio: 0.7
```

`mode: field` (and `bob`) double the output frame rate (e.g. 50i -> 50p) to preserve full motion
-- the stage reports this new rate so later stages (including the final encode) use it correctly.
With `field_order: auto`, the stage runs ffmpeg's `idet` filter first and only deinterlaces when a
strong majority of classified frames are interlaced; otherwise it skips and logs why (visible with
`--verbose` or in the JSON run report). If you already know the field order, set it explicitly
(`tff`/`bff`) to skip detection entirely.

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

### Choosing an Audio Speed Method

When you slow down or speed up a video (`stages.speed.factor` / a future `--speed` flag), the
audio has to change speed too, and there are three ways to do that
(`stages.speed.audio_method` / `--audio-speed-method`):

- **`atempo`** (default) -- keeps the original pitch. Good for normal speed-ups/slow-downs, but at
  extreme factors (e.g. 4x slow-motion) it can sound a bit robotic/artifacted.
- **`rubberband`** -- also keeps pitch, but sounds noticeably cleaner than `atempo` at extreme
  factors. Requires an FFmpeg build with librubberband support; if yours doesn't have it, Auto
  Video Fixer automatically falls back to `atempo` and logs a warning.
- **`asetrate`** -- deliberately changes pitch along with speed. This is the right choice, not a
  workaround, if your clip is **phone slow-motion footage**: phones record slow-mo audio at a
  higher microphone sample rate and then map it down to play at normal speed, so by the time you
  speed the clip back up, a pitch-preserving method leaves the audio pitched too low. `asetrate`
  undoes that mapping and restores the voice/sound to how it actually sounded when recorded.

```yaml
stages:
  speed:
    enabled: true
    factor: 0.25          # 4x slow-motion
    audio_method: asetrate # atempo | rubberband | asetrate
    audio_sample_rate: 48000
    resampler: soxr        # soxr | swr
```

```bash
avf process slowmo.mp4 --enable-stage speed --set stages.speed.factor=0.25 \
  --audio-speed-method asetrate
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

### Symlinks When Scanning Directories

When a directory argument (to `process`, `analyze`, or `find-duplicates`) is scanned for video
files, `general.symlinks` / `--symlinks` controls how symlinks are treated:

- `files_only` (default): a symlinked *file* is picked up like any other file, but a symlinked
  *directory* is never descended into. This is Auto Video Fixer's original behavior, now
  documented rather than silent.
- `follow`: also descend into symlinked directories. Loops (`a -> b -> a`) are detected and
  refused rather than hanging. If the same file is reachable both directly and through a symlink,
  it may be scanned twice -- the existing per-run duplicate-input detection collapses that back
  down to a single job.
- `skip`: ignore every symlink entirely, whether it points at a file or a directory.

```bash
# Also walk into symlinked subdirectories
avf process ./library/ -r --symlinks follow

# Never touch anything reached through a symlink
avf process ./library/ -r --symlinks skip
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
# Use verbose flag
avf --verbose process video.mp4

# Or set log level
avf --log-level DEBUG process video.mp4

# Or log to file
avf --log-file /tmp/avf.log process video.mp4
```

Or set in config:
```yaml
general:
  log_level: DEBUG
```

### API and VLM Integration

Model connections are **endpoints**: named entries under `analysis.endpoints`, each with its own
URL, key, timeout, TLS and request shape. `analysis.vlm.endpoint` picks the one the VLM uses
(`analysis.llm.endpoint` for the scene coordinator), or choose per run with
`avf analyze --vlm-endpoint NAME`.

```yaml
analysis:
  endpoints:
    default_vlm:
      provider: ollama      # ollama | openai | api | llamacpp
      model: llava
      api_url: http://localhost:11434
      api_key: ""
      timeout_sec: 120      # null or 0 = no timeout
  vlm:
    enabled: true
    endpoint: default_vlm
```

**llama.cpp (`llama-server`) setup** -- add an endpoint and select it:
```yaml
analysis:
  endpoints:
    llamacpp_lan:
      provider: llamacpp
      model: your-vision-model
      api_url: "http://192.0.2.10:8080/v1/chat/completions"
      allow_http: true       # plain HTTP to a non-loopback LAN host
      timeout_sec: 600       # big vision models are slow
      compat:
        extra_body: {cache_prompt: true}
  vlm:
    enabled: true
    endpoint: llamacpp_lan
```
`llamacpp` sends the system message as content parts and no `response_format` (what llama.cpp's
chat templates need); override with `compat.system_content` (`string|parts`) and
`compat.response_format` (`json_object|json_schema|none`). Reasoning models' `<think>` blocks and
chatter around the JSON are stripped automatically.

#### How many frames are sent, and how big

The same sampler serves `avf analyze` (whole video) and scene mode (`scenes.drop_non_content`, one
call per scene). For each video or scene ("the range"):

1. **Count**: `ceil(duration / interval_sec)`, clamped to `[min_frames, max_frames]` (defaults
   4 and 8, interval 10 s), then to the endpoint's `vision.max_frames_per_call`. A clip with fewer
   distinct frames than that sends only those (a 3-frame clip sends 3; copies are never added).
2. **Positions**: the centre of each of that many equal slices of the range, found by timestamp,
   so variable-frame-rate files sample the right moments.
3. **Duplicates**: a frame that looks the same as the previous kept one
   (`dedupe_consecutive`, `dedupe_threshold`) is dropped and the count is refilled from the widest
   gap between the kept frames. A -> B -> A is kept; a genuinely static clip may send fewer than
   `min_frames`.
4. **Size**: frames are only ever shrunk to the endpoint's `vision` limits (never enlarged) and
   sent as JPEG at `vision.jpeg_quality` (default 90).

`avf analyze` prints what was sent: *VLM Frames Sent*, *Duplicates Dropped*, *Frame Size* and, if
the endpoint has a tile size, *Est. Tiles / Tokens* (also in the log and as `vlm_*` CSV columns;
scene mode logs it per scene and adds it to the `--report-json` under `scene_stats.vlm_sampling`).

**Telling AVF what your model can digest** (`analysis.endpoints.<name>.vision`, all optional):

```yaml
analysis:
  endpoints:
    llamacpp_lan:
      provider: llamacpp
      # ...
      vision:
        max_long_side_px: 720     # shrink frames to 720 px on the long side (llama.cpp setups)
        tile_width: 448           # the model's vision tile size; enables the token estimate
        tile_height: 448
        max_tiles_per_call: 24    # total image tiles the server/context allows per request
        max_frames_per_call: 8
        jpeg_quality: 90
  vlm:
    sampling:
      min_frames: 4
      max_frames: 8
      balance: auto               # auto | temporal | spatial | fixed
```

- `max_long_side_px` / `max_megapixels` cap the picture size; they work without any tile info.
- With `tile_width`/`tile_height`, a frame costs `ceil(W/tile_w) * ceil(H/tile_h)` tiles (models pad
  each edge up to whole tiles) and `tokens_per_tile` tokens each; `max_tiles_per_frame` and
  `max_tiles_per_call` (needs a tile size) then shrink frames, or trade frames for size, to fit.
- **`balance`** decides that trade when `max_tiles_per_call` is set: `temporal` favours more,
  smaller frames (never below `min_long_side_px`, default 336, unless needed to still send
  `min_frames`; never below 64 px), `spatial` favours fewer, larger ones, `fixed` keeps the plain
  frame count and only shrinks the picture, `auto` looks at how much the video moves and how long
  it is (long, busy ranges get more, smaller frames).
- If hard limits conflict, the order is: `max_tiles_per_call` / `max_frames_per_call` first, then
  `min_frames`, then `min_long_side_px`.
- Per run: `avf analyze --min-sample-frames N --max-sample-frames N --sample-interval SECONDS`.
- Repeats that survived heavy compression (very low bitrate) may need `dedupe_threshold: 0.004`;
  `0` drops only frames that decode identically.

### Config versioning, migration and first run

Your `config.yaml` carries a `config_version`. When AVF is upgraded and your default-path config
is older, the first run **backs it up** (`config.yaml.bak-v1-YYYYMMDD-HHMMSS`, same folder),
rewrites it in place with comments preserved, and prints a WARNING listing every change (crop is
turned on, `normalize_codecs` added to a pinned stage order, VLM/LLM settings moved into
`analysis.endpoints`, sampling keys renamed). Configs passed with `--config`/`AVF_CONFIG` are only
migrated in memory -- run `avf config migrate PATH` (or `-o NEW.yaml`) to update the file. With no
config at all, the first run creates one from the commented example (set
`AVF_NO_CONFIG_BOOTSTRAP=1` to prevent that).

### Corporate Networks and Self-Signed Servers (TLS)

AVF verifies HTTPS certificates for everything it fetches: model downloads and VLM/LLM endpoints.
If that fails with `CERTIFICATE_VERIFY_FAILED`, pick the **narrowest** fix below.

**1. Your company's firewall inspects TLS (model downloads fail at work).** The firewall re-signs
traffic with a private CA. Ask IT for that CA certificate (PEM format), or export it from your
browser's certificate viewer, and trust it *in addition to* the normal system roots:

```yaml
network:
  tls:
    mode: verify
    ca_files: [/etc/ssl/certs/acme-corp-inspection-ca.pem]
```

**2. A self-signed server (e.g. your own llama.cpp/Ollama box behind nginx).** Trust that exact
certificate; AVF still checks that it matches the host name and is in date:

```bash
openssl s_client -connect box.lan:8443 -showcerts </dev/null | openssl x509 > ~/certs/box.pem
```
```yaml
network:
  tls:
    ca_files: [~/certs/box.pem]
```

**3. You cannot get the certificate.** `mode: untrusted_ca` accepts a certificate whose chain does
not verify (self-signed or unknown CA) but still rejects an expired or not-yet-valid certificate
and one issued for a different host name. It is weaker than option 1/2: anyone on the network
path could present their own certificate for the right name. `mode: insecure` turns verification
off entirely (traffic is still encrypted); use it only as a last resort. Both log a WARNING once
per run. Revocation (OCSP/CRL) is not checked in any mode.

**Downloads only.** `network.download_tls` takes the same keys and overrides `network.tls` for
model downloads, e.g. relax TLS for the proxy that intercepts the model hosts while keeping your
VLM endpoint strict. Downloaded models are SHA256-verified in every mode, so even `insecure`
cannot swap in tampered weights for a registry model (a custom `--url` model has no pinned hash).

**Setting it for one command.**

| Command | How |
|---|---|
| `avf process` | `--set network.tls.mode=insecure` (also `--set network.download_tls.mode=untrusted_ca`) |
| `avf analyze`, `avf model-download` | config file only: edit `config.yaml`, or use a throwaway one: `avf --config tls.yaml analyze clip.mp4 --vlm` |

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
