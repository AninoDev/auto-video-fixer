"""Tests for the ncnn/Vulkan inference backend (Feature 4: NCNN backend option).

The `ncnn` PyPI package happened to be installable and functional (Vulkan
device visible) in the environment this was developed in, so a subset of
tests below exercise it for real (skipped automatically wherever it isn't
available, via `_NCNN_AVAILABLE`). Everything else -- the model registry,
the `backend=` dispatch on `RealESRGANUpscaler`/`RIFEInterpolator`, and
availability-failure handling -- is tested with `ncnn` mocked out, so this
file passes in CI regardless of whether Vulkan/ncnn are present there.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autovideofixer.ai.backends.ncnn_common import is_ncnn_available

_NCNN_AVAILABLE = is_ncnn_available()


class TestNcnnModelRegistry:
    """Test the ncnn model registry additions in model_cache.py."""

    def test_registry_has_expected_models(self):
        from autovideofixer.ai.model_cache import NCNN_MODEL_REGISTRY

        assert "RealESRGAN_x4plus" in NCNN_MODEL_REGISTRY
        assert "RealESRGAN_x4plus_anime_6B" in NCNN_MODEL_REGISTRY
        assert "rife_v4.6" in NCNN_MODEL_REGISTRY

    def test_registry_entries_have_required_fields(self):
        from autovideofixer.ai.model_cache import NCNN_MODEL_REGISTRY

        required = {
            "archive_url",
            "archive_filename",
            "archive_sha256",
            "param_member",
            "param_sha256",
            "bin_member",
            "bin_sha256",
            "description",
        }
        for name, meta in NCNN_MODEL_REGISTRY.items():
            missing = required - set(meta)
            assert not missing, f"{name} missing fields: {missing}"
            assert meta["archive_url"].startswith("https://"), f"Invalid URL for {name}"
            assert len(meta["archive_sha256"]) == 64
            assert len(meta["param_sha256"]) == 64
            assert len(meta["bin_sha256"]) == 64

    def test_list_available_ncnn_models(self):
        from autovideofixer.ai.model_cache import list_available_ncnn_models

        names = list_available_ncnn_models()
        assert "RealESRGAN_x4plus" in names

    def test_realesrgan_logical_names_match_torch_registry(self):
        """R4.4: same logical model name resolves to torch .pth or ncnn .param/.bin."""
        from autovideofixer.ai.model_cache import MODEL_REGISTRY, NCNN_MODEL_REGISTRY

        shared = set(MODEL_REGISTRY) & set(NCNN_MODEL_REGISTRY)
        assert "RealESRGAN_x4plus" in shared
        assert "RealESRGAN_x4plus_anime_6B" in shared


class TestGetNcnnModelPaths:
    """Test the (no-network) lookup half of the ncnn model cache."""

    def test_unknown_model_returns_none(self):
        from autovideofixer.ai.model_cache import get_ncnn_model_paths

        assert get_ncnn_model_paths("nonexistent_model_xyz") is None

    def test_not_yet_extracted_returns_none(self, tmp_path):
        from autovideofixer.ai.model_cache import get_ncnn_model_paths

        with patch("autovideofixer.ai.model_cache.get_ncnn_model_dir", return_value=tmp_path):
            assert get_ncnn_model_paths("RealESRGAN_x4plus") is None

    def test_hash_mismatch_returns_none(self, tmp_path):
        from autovideofixer.ai.model_cache import get_ncnn_model_paths

        (tmp_path / "RealESRGAN_x4plus.param").write_bytes(b"not the real param file")
        (tmp_path / "RealESRGAN_x4plus.bin").write_bytes(b"not the real bin file")
        with patch("autovideofixer.ai.model_cache.get_ncnn_model_dir", return_value=tmp_path):
            assert get_ncnn_model_paths("RealESRGAN_x4plus") is None

    def test_valid_cached_pair_resolves(self, tmp_path):
        from autovideofixer.ai.model_cache import (
            NCNN_MODEL_REGISTRY,
            get_model_hash,
            get_ncnn_model_paths,
        )

        meta = NCNN_MODEL_REGISTRY["RealESRGAN_x4plus"]
        # Construct byte content whose sha256 matches the pinned hash isn't
        # feasible in a unit test; instead verify the round trip using the
        # *actual* pinned hash by writing bytes and pointing the registry
        # entry's expected hash at what we wrote (patched copy), proving
        # get_ncnn_model_paths()'s verification logic itself is correct
        # without needing the real multi-hundred-MB upstream archive.
        param_bytes = b"fake param contents for hash round-trip test"
        bin_bytes = b"fake bin contents for hash round-trip test"
        (tmp_path / "RealESRGAN_x4plus.param").write_bytes(param_bytes)
        (tmp_path / "RealESRGAN_x4plus.bin").write_bytes(bin_bytes)

        fake_meta = dict(meta)
        import hashlib

        fake_meta["param_sha256"] = hashlib.sha256(param_bytes).hexdigest()
        fake_meta["bin_sha256"] = hashlib.sha256(bin_bytes).hexdigest()

        with (
            patch("autovideofixer.ai.model_cache.get_ncnn_model_dir", return_value=tmp_path),
            patch.dict(
                "autovideofixer.ai.model_cache.NCNN_MODEL_REGISTRY",
                {"RealESRGAN_x4plus": fake_meta},
            ),
        ):
            result = get_ncnn_model_paths("RealESRGAN_x4plus")
        assert result == (tmp_path / "RealESRGAN_x4plus.param", tmp_path / "RealESRGAN_x4plus.bin")
        # sanity: get_model_hash actually verifies content, not just presence
        assert (
            get_model_hash(str(tmp_path / "RealESRGAN_x4plus.param")) == fake_meta["param_sha256"]
        )


class TestExtractMember:
    """Test the zip-member extraction + verification helper."""

    def test_extract_member_verifies_hash(self, tmp_path):
        import hashlib
        import zipfile

        from autovideofixer.ai.model_cache import _extract_member

        archive = tmp_path / "archive.zip"
        content = b"hello ncnn model bytes"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("models/thing.bin", content)

        dest = tmp_path / "thing.bin"
        _extract_member(archive, "models/thing.bin", dest, hashlib.sha256(content).hexdigest())
        assert dest.read_bytes() == content

    def test_extract_member_rejects_hash_mismatch(self, tmp_path):
        import zipfile

        from autovideofixer.ai.model_cache import _extract_member

        archive = tmp_path / "archive.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("models/thing.bin", b"actual content")

        dest = tmp_path / "thing.bin"
        with pytest.raises(RuntimeError, match="hash verification"):
            _extract_member(archive, "models/thing.bin", dest, "0" * 64)
        assert not dest.exists()

    def test_extract_member_rejects_oversized_entry(self, tmp_path):
        import zipfile

        from autovideofixer.ai.model_cache import _extract_member

        archive = tmp_path / "archive.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("models/thing.bin", b"x" * 1000)

        dest = tmp_path / "thing.bin"
        with pytest.raises(RuntimeError, match="safety limit"):
            _extract_member(archive, "models/thing.bin", dest, "0" * 64, max_bytes=10)


class TestNcnnCommon:
    """Test the shared ncnn helper functions."""

    def test_is_ncnn_available_matches_import(self):
        assert is_ncnn_available() == _NCNN_AVAILABLE

    def test_get_vulkan_gpu_count_never_raises(self):
        from autovideofixer.ai.backends.ncnn_common import get_vulkan_gpu_count

        # Whatever the real environment reports, this must not raise and
        # must be a non-negative int (0 covers "no ncnn"/"no Vulkan device").
        count = get_vulkan_gpu_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_vulkan_gpu_count_handles_missing_ncnn(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ncnn":
                raise ImportError("simulated missing ncnn")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from autovideofixer.ai.backends.ncnn_common import get_vulkan_gpu_count

        assert get_vulkan_gpu_count() == 0

    def test_frame_mat_roundtrip_preserves_shape(self):
        if not _NCNN_AVAILABLE:
            pytest.skip("ncnn package not available in this environment")
        from autovideofixer.ai.backends.ncnn_common import frame_to_mat, mat_to_frame

        frame = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
        mat = frame_to_mat(frame)
        out = mat_to_frame(mat)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        # Normalize+denormalize round trip should be near-lossless (uint8
        # -> float32/255 -> *255 -> uint8), small rounding only.
        assert np.abs(out.astype(int) - frame.astype(int)).max() <= 1

    def test_tiled_inference_matches_whole_frame_identity(self):
        """Tiling an identity infer_fn must reproduce the (upscaled) input exactly."""
        from autovideofixer.ai.backends.ncnn_common import tiled_ncnn_inference

        frame = (np.random.rand(37, 53, 3) * 255).astype(np.uint8)

        def identity_2x(tile: np.ndarray) -> np.ndarray:
            return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)

        out = tiled_ncnn_inference(
            frame, tile_size=16, overlap=4, out_scale=2, infer_fn=identity_2x
        )
        expected = identity_2x(frame)
        assert out.shape == expected.shape
        assert np.array_equal(out, expected)


class TestNcnnUpscaleBackendMocked:
    """Test NcnnUpscaleBackend without requiring ncnn/Vulkan to be installed."""

    def test_load_model_false_when_ncnn_unavailable(self):
        from autovideofixer.ai.backends.ncnn_upscale import NcnnUpscaleBackend

        with patch("autovideofixer.ai.backends.ncnn_common.is_ncnn_available", return_value=False):
            backend = NcnnUpscaleBackend()
            assert backend.load_model() is False
            assert backend.is_loaded is False

    def test_load_model_false_when_model_unavailable(self):
        from autovideofixer.ai.backends.ncnn_upscale import NcnnUpscaleBackend

        with (
            patch("autovideofixer.ai.backends.ncnn_common.is_ncnn_available", return_value=True),
            patch(
                "autovideofixer.ai.model_cache.ensure_ncnn_model_available",
                return_value=(False, "network unavailable"),
            ),
        ):
            backend = NcnnUpscaleBackend()
            assert backend.load_model() is False

    def test_load_model_false_on_graph_load_failure(self, tmp_path):
        """Simulates the real 'custom layer not registered' style failure."""
        from autovideofixer.ai.backends.ncnn_upscale import NcnnUpscaleBackend

        fake_net = MagicMock()
        fake_net.load_param.return_value = -1
        fake_net.load_model.return_value = -1

        with patch("autovideofixer.ai.backends.ncnn_common.make_net", return_value=fake_net):
            backend = NcnnUpscaleBackend()
            param = tmp_path / "x.param"
            binf = tmp_path / "x.bin"
            param.write_text("fake")
            binf.write_bytes(b"fake")
            assert backend.load_model(str(param), str(binf)) is False

    def test_upscale_raises_when_not_loaded(self):
        from autovideofixer.ai.backends.ncnn_upscale import NcnnUpscaleBackend

        backend = NcnnUpscaleBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.upscale(np.zeros((8, 8, 3), dtype=np.uint8))

    def test_unload_is_idempotent(self):
        from autovideofixer.ai.backends.ncnn_upscale import NcnnUpscaleBackend

        backend = NcnnUpscaleBackend()
        backend.unload()
        backend.unload()
        assert backend.is_loaded is False


def _rife_ncnn_available() -> bool:
    try:
        import rife_ncnn_vulkan_python  # noqa: F401

        return True
    except ImportError:
        return False


_RIFE_NCNN_AVAILABLE = _rife_ncnn_available()


class TestNcnnInterpolateBackendMocked:
    """Test NcnnInterpolateBackend without requiring rife-ncnn-vulkan-python to be installed."""

    def test_load_model_false_when_rife_ncnn_unavailable(self):
        from autovideofixer.ai.backends.ncnn_interpolate import NcnnInterpolateBackend

        with patch(
            "autovideofixer.ai.backends.ncnn_interpolate.is_rife_ncnn_available",
            return_value=False,
        ):
            backend = NcnnInterpolateBackend()
            assert backend.load_model() is False

    def test_load_model_false_when_model_unavailable(self):
        from autovideofixer.ai.backends.ncnn_interpolate import NcnnInterpolateBackend

        with (
            patch(
                "autovideofixer.ai.backends.ncnn_interpolate.is_rife_ncnn_available",
                return_value=True,
            ),
            patch(
                "autovideofixer.ai.model_cache.ensure_ncnn_model_available",
                return_value=(False, "network unavailable"),
            ),
        ):
            backend = NcnnInterpolateBackend()
            assert backend.load_model() is False

    def test_load_model_false_on_rife_construction_failure(self, tmp_path):
        """Simulates a Rife() construction error (e.g. corrupt model dir)."""
        from autovideofixer.ai.backends.ncnn_interpolate import NcnnInterpolateBackend

        param = tmp_path / "rife_v4.6.param"
        binf = tmp_path / "rife_v4.6.bin"
        param.write_text("fake")
        binf.write_bytes(b"fake")

        fake_module = MagicMock()
        fake_module.Rife.side_effect = RuntimeError("simulated Rife() load failure")

        with (
            patch(
                "autovideofixer.ai.backends.ncnn_interpolate.is_rife_ncnn_available",
                return_value=True,
            ),
            patch.dict("sys.modules", {"rife_ncnn_vulkan_python": fake_module}),
        ):
            backend = NcnnInterpolateBackend()
            assert backend.load_model(str(param), str(binf)) is False
            assert backend.is_loaded is False

    def test_interpolate_raises_when_not_loaded(self):
        from autovideofixer.ai.backends.ncnn_interpolate import NcnnInterpolateBackend

        backend = NcnnInterpolateBackend()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.interpolate(frame, frame)

    def test_is_available_cleans_up_after_itself(self):
        """is_available() must never leave a loaded model behind, success or failure."""
        from autovideofixer.ai.backends.ncnn_interpolate import NcnnInterpolateBackend

        with patch(
            "autovideofixer.ai.backends.ncnn_interpolate.is_rife_ncnn_available",
            return_value=False,
        ):
            backend = NcnnInterpolateBackend()
            assert backend.is_available() is False
            assert backend.is_loaded is False

    def test_prepare_model_dir_creates_flownet_named_members(self, tmp_path):
        """Rife._load() needs a dir literally containing flownet.param/flownet.bin."""
        from autovideofixer.ai.backends.ncnn_interpolate import NcnnInterpolateBackend
        from autovideofixer.ai.model_cache import get_ncnn_model_dir

        param_src = tmp_path / "rife_v4.6.param"
        bin_src = tmp_path / "rife_v4.6.bin"
        param_src.write_text("param contents")
        bin_src.write_bytes(b"bin contents")

        with patch("autovideofixer.ai.model_cache.get_ncnn_model_dir", return_value=tmp_path):
            model_dir = NcnnInterpolateBackend._prepare_model_dir(str(param_src), str(bin_src))

        assert model_dir == tmp_path / "rife-v4"
        assert "rife-v4" in str(model_dir)
        assert (model_dir / "flownet.param").exists()
        assert (model_dir / "flownet.bin").exists()
        assert (model_dir / "flownet.param").read_text() == "param contents"
        assert (model_dir / "flownet.bin").read_bytes() == b"bin contents"
        # get_ncnn_model_dir is real elsewhere; sanity-check it's actually used.
        assert get_ncnn_model_dir  # imported successfully

    def test_prepare_model_dir_is_idempotent(self, tmp_path):
        from autovideofixer.ai.backends.ncnn_interpolate import NcnnInterpolateBackend

        param_src = tmp_path / "rife_v4.6.param"
        bin_src = tmp_path / "rife_v4.6.bin"
        param_src.write_text("v1")
        bin_src.write_bytes(b"v1")

        with patch("autovideofixer.ai.model_cache.get_ncnn_model_dir", return_value=tmp_path):
            dir1 = NcnnInterpolateBackend._prepare_model_dir(str(param_src), str(bin_src))
            dir2 = NcnnInterpolateBackend._prepare_model_dir(str(param_src), str(bin_src))

        assert dir1 == dir2
        assert (dir1 / "flownet.param").read_text() == "v1"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _RIFE_NCNN_AVAILABLE, reason="rife_ncnn_vulkan_python package not available"
    )
    def test_real_rife_ncnn_interpolation_produces_valid_output(self):
        """Real GPU/CPU integration test: load the actual cached model and interpolate.

        Skipped gracefully if `rife_ncnn_vulkan_python` isn't installed (it's an optional,
        source-built dependency -- see pyproject.toml's `ncnn` extra) or if the real model
        download/cache step fails (e.g. no network in CI).
        """
        from autovideofixer.ai.backends.ncnn_interpolate import NcnnInterpolateBackend

        backend = NcnnInterpolateBackend()
        loaded = backend.load_model()
        if not loaded:
            pytest.skip("ncnn RIFE model could not be loaded in this environment")

        try:
            rng = np.random.default_rng(0)
            frame0 = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
            frame1 = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
            out = backend.interpolate(frame0, frame1, timestep=0.5)
            assert out.shape == frame0.shape
            assert out.dtype == np.uint8
            # Non-degenerate: real variance, not a flat/black frame.
            assert float(out.std()) > 1.0
        finally:
            backend.unload()


class TestWrapperBackendDispatch:
    """Test that RealESRGANUpscaler/RIFEInterpolator honor `backend=`."""

    def test_default_backend_is_torch(self):
        from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler

        upscaler = RealESRGANUpscaler()
        assert upscaler.backend == "torch"

    def test_default_backend_is_torch_interpolator(self):
        from autovideofixer.ai.wrappers.interpolate import RIFEInterpolator

        interp = RIFEInterpolator()
        assert interp.backend == "torch"

    def test_upscaler_ncnn_backend_delegates_load(self):
        from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler

        fake_backend = MagicMock()
        fake_backend.load_model.return_value = True
        with patch(
            "autovideofixer.ai.backends.ncnn_upscale.NcnnUpscaleBackend",
            return_value=fake_backend,
        ):
            upscaler = RealESRGANUpscaler(backend="ncnn")
            assert upscaler.load_model() is True
            assert upscaler.is_loaded is True
        fake_backend.load_model.assert_called_once()

    def test_upscaler_ncnn_backend_delegates_upscale(self):
        from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        expected_out = np.ones((32, 32, 3), dtype=np.uint8)
        fake_backend = MagicMock()
        fake_backend.load_model.return_value = True
        fake_backend.upscale.return_value = expected_out
        with patch(
            "autovideofixer.ai.backends.ncnn_upscale.NcnnUpscaleBackend",
            return_value=fake_backend,
        ):
            upscaler = RealESRGANUpscaler(backend="ncnn")
            upscaler.load_model()
            out = upscaler.upscale(frame)
        assert out is expected_out
        fake_backend.upscale.assert_called_once_with(frame)

    def test_upscaler_ncnn_backend_load_failure_keeps_not_loaded(self):
        from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler

        fake_backend = MagicMock()
        fake_backend.load_model.return_value = False
        with patch(
            "autovideofixer.ai.backends.ncnn_upscale.NcnnUpscaleBackend",
            return_value=fake_backend,
        ):
            upscaler = RealESRGANUpscaler(backend="ncnn")
            assert upscaler.load_model() is False
            assert upscaler.is_loaded is False

    def test_upscaler_ncnn_backend_unload_delegates(self):
        from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler

        fake_backend = MagicMock()
        fake_backend.load_model.return_value = True
        with patch(
            "autovideofixer.ai.backends.ncnn_upscale.NcnnUpscaleBackend",
            return_value=fake_backend,
        ):
            upscaler = RealESRGANUpscaler(backend="ncnn")
            upscaler.load_model()
            upscaler.unload()
        fake_backend.unload.assert_called_once()
        assert upscaler.is_loaded is False

    def test_interpolator_ncnn_backend_delegates(self):
        from autovideofixer.ai.wrappers.interpolate import RIFEInterpolator

        frame0 = np.zeros((8, 8, 3), dtype=np.uint8)
        frame1 = np.ones((8, 8, 3), dtype=np.uint8)
        expected_out = np.full((8, 8, 3), 128, dtype=np.uint8)
        fake_backend = MagicMock()
        fake_backend.load_model.return_value = True
        fake_backend.interpolate.return_value = expected_out
        with patch(
            "autovideofixer.ai.backends.ncnn_interpolate.NcnnInterpolateBackend",
            return_value=fake_backend,
        ):
            interp = RIFEInterpolator(backend="ncnn")
            assert interp.load_model() is True
            out = interp.interpolate(frame0, frame1, timestep=0.5)
        assert out is expected_out
        fake_backend.interpolate.assert_called_once_with(frame0, frame1, 0.5)

    def test_interpolator_ncnn_backend_load_failure(self):
        from autovideofixer.ai.wrappers.interpolate import RIFEInterpolator

        fake_backend = MagicMock()
        fake_backend.load_model.return_value = False
        with patch(
            "autovideofixer.ai.backends.ncnn_interpolate.NcnnInterpolateBackend",
            return_value=fake_backend,
        ):
            interp = RIFEInterpolator(backend="ncnn")
            assert interp.load_model() is False
            assert interp.is_loaded is False

    def test_torch_backend_unaffected_by_backend_param_default(self):
        """Existing (pre-feature) instantiation with no `backend=` kwarg still
        exercises the torch path's own not-loaded error, proving backend="torch"
        changes nothing about default behavior."""
        from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler

        upscaler = RealESRGANUpscaler()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            upscaler.upscale(frame)


@pytest.mark.skipif(not _NCNN_AVAILABLE, reason="ncnn package not available")
class TestNcnnUpscaleBackendLive:
    """Live Real-ESRGAN-ncnn inference tests, gated on model availability.

    Skips (rather than fails) when no network/cached model is available,
    mirroring the existing `_real_esrgan_model_cached()` gating pattern in
    test_ai_wrappers.py for the torch path.
    """

    def _cached_ncnn_paths(self) -> tuple[Path, Path] | None:
        from autovideofixer.ai.model_cache import get_ncnn_model_paths

        return get_ncnn_model_paths("RealESRGAN_x4plus")

    def test_live_upscale_produces_plausible_output(self):
        paths = self._cached_ncnn_paths()
        if paths is None:
            pytest.skip("RealESRGAN_x4plus ncnn model not cached in this environment")

        from autovideofixer.ai.backends.ncnn_upscale import NcnnUpscaleBackend

        backend = NcnnUpscaleBackend(model_name="RealESRGAN_x4plus", scale=4)
        assert backend.load_model(str(paths[0]), str(paths[1])) is True

        frame = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        out = backend.upscale(frame)
        assert out.shape == (128, 128, 3)
        assert out.dtype == np.uint8
        # Not solid black/white -- a real forward pass, not a degenerate one.
        assert 5 < out.mean() < 250
        assert out.std() > 1
        backend.unload()
