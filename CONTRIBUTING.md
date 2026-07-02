# Contributing to Auto Video Fixer

Thank you for your interest in contributing to Auto Video Fixer! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Adding a New Stage](#adding-a-new-stage)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Documentation](#documentation)

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. Be respectful, inclusive, and supportive.

---

## Getting Started

### Prerequisites

- Python 3.14 or higher
- FFmpeg (installed and in PATH)
- Git
- uv (package manager)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/auto-video-fixer.git
   cd auto-video-fixer
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/originalusername/auto-video-fixer.git
   ```

### Development Setup

```bash
# Create virtual environment
uv venv

# Activate
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install with development dependencies
uv pip install -e ".[all]"

# Verify installation
avf --version
```

---

## Project Structure

```
auto-video-fixer/
├── src/autovideofixer/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── logger.py              # Logging utilities
│   ├── core/
│   │   ├── pipeline.py        # Pipeline orchestrator
│   │   ├── quality.py         # VMAF quality estimation
│   │   ├── analysis.py        # Video analysis
│   │   ├── ffmpeg_utils.py    # FFmpeg integration
│   │   ├── presets.py         # Processing presets
│   │   └── stages/            # Processing stages
│   │       ├── base.py        # Base stage class
│   │       ├── detect.py
│   │       ├── stabilize.py
│   │       └── ...
│   ├── gui/
│   │   └── main_window.py     # Qt GUI
│   └── cli/
│       └── cli.py             # CLI interface
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── conftest.py           # Test fixtures
├── docs/                      # Documentation
├── pyproject.toml            # Project configuration
└── README.md                 # User documentation
```

---

## Adding a New Stage

Adding a new processing stage is straightforward thanks to the modular architecture.

### Step 1: Create Stage File

Create a new file in `src/autovideofixer/core/stages/`:

```python
# src/autovideofixer/core/stages/my_stage.py
"""Auto Video Fixer - My Custom Stage."""

from __future__ import annotations

import time
from typing import Any

from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class MyCustomStage(BaseStage):
    """Custom processing stage.
    
    This stage does something cool to videos.
    """
    
    name = "my_custom"
    display_name = "My Custom Stage"
    description = "Does something awesome"
    category = "enhancement"  # analysis, enhancement, encoding, output
    priority = 25  # Lower = runs earlier
    supports_gpu = True  # Set True if using GPU
    supports_hardware_encoding = False
    
    def __init__(self, config):
        super().__init__(config)
        # Load stage-specific config
        self._stage_config = config.get("stages", self.name, default={})
        self._param = self._stage_config.get("param", "default")
    
    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        """Determine if this stage should run.
        
        Args:
            input_info: Video metadata from detect stage
            
        Returns:
            (should_run, reason) tuple
        """
        if not self.is_enabled():
            return False, "Stage disabled in configuration"
        
        # Add your logic here
        # Example: skip if resolution is already high enough
        w, h = input_info.get("resolution", (0, 0))
        if w >= 3840 and h >= 2160:
            return False, "Already 4K or higher"
        
        return True, None
    
    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        **kwargs,
    ) -> StageResult:
        """Execute the processing stage.
        
        Args:
            input_path: Path to input file
            output_path: Path for output file
            progress_callback: Function(progress: float, message: str)
            **kwargs: Additional parameters
            
        Returns:
            StageResult with outcome
        """
        start_time = time.time()
        
        self._report_progress(0.0, "Starting my custom stage...", progress_callback)
        
        try:
            # Your processing logic here
            # Example using FFmpeg:
            from autovideofixer.core.ffmpeg_utils import get_ffmpeg_path, run_ffmpeg
            
            ffmpeg = get_ffmpeg_path()
            args = [
                "-i", input_path,
                "-vf", "myfilter=param={}".format(self._param),
                "-c:a", "copy",
                "-y", output_path
            ]
            
            def progress_cb(p, m):
                self._report_progress(0.3 + p * 0.7, m, progress_callback)
            
            result = run_ffmpeg(args, progress_callback=progress_cb, timeout=600)
            
            if result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"My stage failed: {result.stderr[:200]}",
                    duration_sec=time.time() - start_time,
                )
            
            self._report_progress(1.0, "My custom stage complete", progress_callback)
            
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={"param": self._param},
                duration_sec=time.time() - start_time,
            )
        
        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start_time,
            )
```

### Step 2: Register the Stage

Add your stage to `src/autovideofixer/core/stages/__init__.py`:

```python
from autovideofixer.core.stages.my_stage import MyCustomStage

# Register as a bare function call (NOT a decorator)
register_stage(MyCustomStage)
```

**Note**: `register_stage()` works as both a decorator (`@register_stage`) and a bare function call (`register_stage(MyStage)`). Both are valid.

### Step 3: Add Stage Config

Add stage configuration to `DEFAULTS["stages"]` in `src/autovideofixer/config.py`:

```python
"stages": {
    "my_custom": {
        "enabled": True,
        "param": "default_value",
    },
    # ... other stages
}
```

**CRITICAL - Stage ordering**: The pipeline does **not** use `DEFAULTS["pipeline"]["default_order"]` from config.py. The actual ordering is hardcoded in `Pipeline.optimize_stage_order()` in `pipeline.py:232`. Changing `DEFAULTS["pipeline"]["default_order"]` has **no effect** on execution order. To change order, modify `pipeline.py:optimize_stage_order()`.

### Step 4: Add Configuration

Add stage configuration to `DEFAULTS["stages"]`:

```python
"stages": {
    "my_custom": {
        "enabled": True,
        "param": "default_value",
    },
    # ... other stages
}
```

### Step 5: Write Tests

Create tests in `tests/unit/test_stages.py`:

```python
def test_my_custom_stage():
    """Test my custom stage."""
    config = Config()
    stage = create_stage("my_custom", config)
    
    assert stage is not None
    assert stage.name == "my_custom"
    
    # Test should_run
    should_run, reason = stage.should_run({"resolution": (1920, 1080)})
    assert should_run is True
    
    # Test execution (add integration test)
```

### Step 6: Run Tests

```bash
# Run all tests
pytest tests/unit/test_stages.py -v

# Run with coverage
pytest --cov=autovideofixer.core.stages.my_stage
```

---

## Running Tests

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_pipeline.py -v

# Run with coverage
pytest --cov=autovideofixer --cov-report=html
```

### Integration Tests

```bash
# Run integration tests (requires FFmpeg)
pytest tests/ -v -m "integration"
```

### Linting

```bash
# Check code style
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type checking
mypy src/ --ignore-missing-imports
```

### Pre-commit Checks

```bash
# Install pre-commit hooks
uv pip install pre-commit
pre-commit install

# Run all hooks
pre-commit run --all-files
```

---

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some additions:

- **Line length**: 100 characters
- **Indentation**: 4 spaces
- **Quotes**: Single quotes for strings, unless contains single quote
- **Imports**: Grouped and sorted (stdlib, third-party, local)
- **Type hints**: Required for all function signatures

### Example

```python
def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> StageResult:
    """Process a video file.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        progress_callback: Optional progress callback
        
    Returns:
        StageResult with processing outcome
    """
    # Your code here
    pass
```

### Docstrings

Use [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-documents) docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """Do something.
    
    Args:
        param1: First parameter
        param2: Second parameter
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        ValueError: If param1 is negative
        TypeError: If param2 is not a string
    """
```

---

## Submitting Changes

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new upscaling stage
fix: resolve deblock filter compatibility
docs: update installation instructions
test: add tests for pipeline ordering
chore: update dependencies
```

### Pull Request Process

1. **Create a branch**:
   ```bash
   git checkout -b feat/my-new-feature
   ```

2. **Make changes**:
   - Write code
   - Add tests
   - Update documentation

3. **Run tests**:
   ```bash
   pytest tests/unit/ -v
   ruff check src/ tests/
   ```

4. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add my new feature"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feat/my-new-feature
   ```
   
   Then create a Pull Request on GitHub.

### PR Guidelines

- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] No breaking changes (or clearly documented)
- [ ] Squash commits if needed

---

## Documentation

### User Documentation

Located in `docs/USER_GUIDE.md`. Update when:
- Adding new features
- Changing behavior
- Fixing bugs

### Developer Documentation

Located in `docs/DEVELOPER.md`. Update when:
- Adding new stages
- Changing architecture
- Adding APIs

### API Documentation

Auto-generated from docstrings. Keep docstrings up to date.

### Building Documentation

```bash
# Install dependencies
uv pip install sphinx

# Build HTML docs
cd docs
sphinx-build -b html . _build/html
```

---

## Reporting Bugs

### Before Reporting

1. **Check existing issues**: Search for similar bugs
2. **Update**: Make sure you're on the latest version
3. **Reproduce**: Create a minimal example
4. **Logs**: Enable debug logging

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run 'avf process video.mp4 -p preset'
2. ...

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.14]
- FFmpeg: [e.g., 5.1]
- Auto Video Fixer: [e.g., 0.1.0]

**Additional context**
Any other context about the problem.
```

---

## Suggesting Features

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
Any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request.
```

---

## Getting Help

- **GitHub Issues**: [Report bugs and features](https://github.com/yourusername/auto-video-fixer/issues)
- **Discussions**: [Community Q&A](https://github.com/yourusername/auto-video-fixer/discussions)
- **Discord**: [Join server](https://discord.gg/yourserver)

---

## Recognition

Contributors are recognized in:
- README.md (top contributors)
- Release notes
- Website (if applicable)

---

*Thank you for contributing to Auto Video Fixer!*
