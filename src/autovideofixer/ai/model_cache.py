"""Auto Video Fixer - AI model download and cache management.

Handles downloading, caching, and checking availability of AI models
(Real-ESRGAN, RIFE, etc.) for video processing stages.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from autovideofixer.config import get_data_dir

_logger: logging.Logger | None = None

# Model names and custom filenames become path components on disk; restrict
# them to a safe charset so a crafted "../../.." name (or one embedding an
# absolute path) can't escape the model cache directory (see download_model).
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ModelPathError(ValueError):
    """Raised when a model name/filename/URL fails safety validation."""


def _validate_safe_name(name: str, what: str) -> None:
    if not name or not _SAFE_NAME_RE.fullmatch(name):
        raise ModelPathError(
            f"Invalid {what} {name!r}: must match {_SAFE_NAME_RE.pattern} "
            "(no path separators or '..')"
        )


def _validate_dest_containment(dest: Path, model_dir: Path) -> None:
    resolved_dest = dest.resolve()
    resolved_dir = model_dir.resolve()
    if not resolved_dest.is_relative_to(resolved_dir):
        raise ModelPathError(
            f"Refusing to write outside model directory: {resolved_dest} not under {resolved_dir}"
        )


def _validate_https_url(url: str) -> None:
    scheme = urlparse(url).scheme.lower()
    if scheme != "https":
        raise ModelPathError(f"Refusing non-https model URL (scheme={scheme!r}): {url}")


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        from autovideofixer.logger import get_logger

        _logger = get_logger("autovideofixer.ai.model_cache")
    return _logger


# Default model URLs and metadata
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "size_mb": 65,
        "description": "Real-ESRGAN x4 upscaling model",
        "scale": 4,
    },
    "RealESRGAN_x2plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "size_mb": 65,
        "description": "Real-ESRGAN x2 upscaling model",
        "scale": 2,
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "size_mb": 19,
        "description": "Real-ESRGAN x4 anime model (lighter, 6 RRDB blocks)",
        "scale": 4,
    },
    # Real weights from the official author's HuggingFace mirror
    # (https://huggingface.co/hzwer/RIFE), verified: the packaged
    # `flownet.pkl` loads into our IFNet with zero missing/unexpected keys.
    # Registry key kept as "rife_v4.6" for config/preset compatibility, but
    # the actual shipped checkpoint is RIFE v4.25/4.26 (internal
    # Model.version=4.25, release archive RIFEv4.26_0921.zip) -- there is
    # no longer a real, separately-downloadable v4.6 or v4.11 checkpoint,
    # so a fabricated second "rife_v4.11" entry pointing at different fake
    # bytes was removed rather than kept as a misleading duplicate.
    "rife_v4.6": {
        "url": "https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip",
        "filename": "rife_v4.6.zip",
        "sha256": "1fa9b9cda3d9b8c3e301359e2595960902f97bf926c08598b0e9957a3f3f760e",
        "size_mb": 22,
        "description": (
            "RIFE v4.25/4.26 frame interpolation model (flownet.pkl, official "
            "hzwer/RIFE HuggingFace mirror, packaged as a release zip)"
        ),
    },
}


def get_model_dir() -> Path:
    """Return the directory where AI models are stored."""
    model_dir = get_data_dir() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_model_path(model_name: str) -> Path | None:
    """Get the cached path for a model, or None if not cached.

    Args:
        model_name: Name of the model (e.g., 'RealESRGAN_x4plus').

    Returns:
        Path to the model file, or None if not found.
    """
    meta = MODEL_REGISTRY.get(model_name)
    if meta is None:
        return None

    model_dir = get_model_dir()
    candidate = model_dir / meta["filename"]
    if candidate.exists():
        expected_sha256 = meta.get("sha256")
        if expected_sha256:
            actual = get_model_hash(str(candidate))
            if actual is None or actual.lower() != expected_sha256.lower():
                _get_logger().warning(
                    f"Cached model {model_name} at {candidate} failed hash "
                    f"re-verification (expected {expected_sha256}, got {actual}); "
                    "treating as not cached, it will be re-downloaded"
                )
                return None
        return candidate

    # Also check models/ directory at project root, for local development
    # only. This bypasses hash verification entirely, so it's opt-in via
    # AVF_ALLOW_DEV_MODELS to avoid silently trusting whatever happens to be
    # in the current working directory's models/ folder.
    if os.environ.get("AVF_ALLOW_DEV_MODELS"):
        project_models = Path("models") / model_name
        if project_models.exists():
            _get_logger().warning(
                f"Loading model {model_name} from unverified dev path "
                f"{project_models} (AVF_ALLOW_DEV_MODELS set)"
            )
            return project_models

    return None


def ensure_model_available(
    model_name: str,
    model_path: str | None = None,
    force_download: bool = False,
) -> tuple[bool, str]:
    """Ensure a model is available, downloading if necessary.

    Args:
        model_name: Name of the model to check/download.
        model_path: Optional custom path to model file.
        force_download: If True, redownload even if cached.

    Returns:
        (success, message) tuple.
    """
    if model_path:
        if os.path.isfile(model_path):
            return True, f"Using custom model at {model_path}"
        return False, f"Custom model path not found: {model_path}"

    cached = get_model_path(model_name)
    if cached and not force_download:
        return True, f"Model already cached at {cached}"

    return download_model(model_name)


def download_model(
    model_name: str,
    url: str | None = None,
    custom_path: str | None = None,
) -> tuple[bool, str]:
    """Download a model from the registry or a custom URL.

    Args:
        model_name: Name of the model.
        url: Optional override URL for the model.
        custom_path: Optional custom filename.

    Returns:
        (success, message) tuple.
    """
    meta = MODEL_REGISTRY.get(model_name)
    if meta is None and url is None:
        return False, f"Unknown model: {model_name}. Available: {list_available_models()}"

    try:
        _validate_safe_name(model_name, "model name")

        expected_sha256: str | None = None
        if url is not None:
            _validate_https_url(url)
            filename = custom_path or f"{model_name}.pth"
            if custom_path:
                _validate_safe_name(custom_path, "custom_path")
        elif meta:
            filename = meta["filename"]
            url = url or meta["url"]
            _validate_https_url(url)
            expected_sha256 = meta.get("sha256")
        else:
            return False, "Must provide URL for custom models"

        model_dir = get_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        dest = model_dir / filename
        _validate_dest_containment(dest, model_dir)
    except ModelPathError as e:
        _get_logger().error(f"Rejected model download request: {e}")
        return False, str(e)

    if expected_sha256 is None:
        _get_logger().warning(
            f"No sha256 pinned for model {model_name!r} in the registry; "
            "downloaded weights will NOT be integrity-verified. Tampered or "
            "corrupted weights would be silently accepted."
        )

    _get_logger().info(f"Downloading {model_name} from {url}")

    try:
        _download_file(url, str(dest), expected_sha256=expected_sha256)
        _get_logger().info(f"Model saved to {dest}")
        return True, f"Downloaded to {dest}"
    except Exception as e:
        _get_logger().error(f"Failed to download model: {e}")
        return False, f"Download failed: {e}"


def _download_file(
    url: str,
    dest: str,
    chunk_size: int = 8192,
    expected_sha256: str | None = None,
) -> None:
    """Download a file from URL to dest path.

    Downloads to a `.part` temp file first and atomically renames into
    place only on success (and only after hash verification, if an
    expected SHA256 was provided). This means an interrupted download or a
    hash mismatch never leaves a corrupt/tampered file at `dest` for
    `get_model_path()` to later treat as validly cached.
    """
    tmp_dest = f"{dest}.part"
    try:
        _fetch_to(url, tmp_dest, chunk_size)

        if expected_sha256:
            actual = get_model_hash(tmp_dest)
            if actual is None or actual.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"Downloaded file hash mismatch for {url} "
                    f"(expected {expected_sha256}, got {actual})"
                )

        os.replace(tmp_dest, dest)
    finally:
        if os.path.exists(tmp_dest):
            try:
                os.unlink(tmp_dest)
            except OSError:
                pass


_DOWNLOAD_TIMEOUT_SEC = 60


def _fetch_to(url: str, dest: str, chunk_size: int = 8192) -> None:
    """Fetch `url` to `dest`, trying urllib, then requests, then curl."""
    _validate_https_url(url)
    errors: list[str] = []

    try:
        import urllib.request

        with (
            urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SEC) as resp,
            open(dest, "wb") as f,
        ):
            while chunk := resp.read(chunk_size):
                f.write(chunk)
        return
    except Exception as e:
        errors.append(f"urllib: {e}")
        _get_logger().warning(f"Download via urllib failed for {url}: {e}")

    try:
        import requests

        resp = requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SEC)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        return
    except ImportError:
        pass
    except Exception as e:
        errors.append(f"requests: {e}")
        _get_logger().warning(f"Download via requests failed for {url}: {e}")

    import subprocess

    result = subprocess.run(
        ["curl", "-fsSL", "--max-time", "300", "-o", dest, url],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        errors.append(f"curl: {result.stderr}")
        raise RuntimeError(f"All download methods failed for {url}: {'; '.join(errors)}")


def list_available_models() -> list[str]:
    """List all known model names in the registry."""
    return list(MODEL_REGISTRY.keys())


def list_cached_models() -> list[dict[str, Any]]:
    """List all models currently cached on disk.

    Returns:
        List of dicts with 'name', 'path', 'size_mb', and 'description'.
    """
    model_dir = get_model_dir()
    cached: list[dict[str, Any]] = []

    for meta_name, meta in MODEL_REGISTRY.items():
        path = model_dir / meta["filename"]
        if path.exists():
            stat = path.stat()
            cached.append(
                {
                    "name": meta_name,
                    "path": str(path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 1),
                    "description": meta.get("description", ""),
                }
            )

    # Also check project-level models/ directory -- gated the same way as
    # get_model_path()'s dev fallback, since these are unverified/unhashed and
    # listing them here without the gate would surface (and implicitly lend
    # trust to) unverified dev models even when AVF_ALLOW_DEV_MODELS is unset.
    if os.environ.get("AVF_ALLOW_DEV_MODELS"):
        project_models = Path("models")
        if project_models.is_dir():
            for item in project_models.iterdir():
                if item.is_file():
                    cached.append(
                        {
                            "name": item.stem,
                            "path": str(item),
                            "size_mb": round(item.stat().st_size / (1024 * 1024), 1),
                            "description": "Custom model (unverified, dev)",
                        }
                    )

    return cached


def get_model_hash(model_path: str) -> str | None:
    """Compute SHA256 hash of a model file for integrity checking."""
    if not os.path.isfile(model_path):
        return None
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def clear_model_cache(model_name: str | None = None) -> int:
    """Remove cached models.

    Args:
        model_name: If provided, remove only this model. Otherwise clear all.

    Returns:
        Number of files removed.
    """
    model_dir = get_model_dir()
    removed = 0

    if model_name:
        meta = MODEL_REGISTRY.get(model_name)
        if meta:
            path = model_dir / meta["filename"]
            if path.exists():
                path.unlink()
                removed = 1
    else:
        if model_dir.exists():
            for f in model_dir.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                        removed += 1
                    except OSError:
                        pass

    return removed
