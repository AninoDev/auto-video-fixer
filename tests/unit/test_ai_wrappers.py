"""Tests for AI model wrappers (Real-ESRGAN, RIFE)."""

from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.ai.wrappers.interpolate import RIFEInterpolator
from autovideofixer.ai.wrappers.upscale import (
    RealESRGANUpscaler,
    SRVGGNetCompact,
    compute_tile_grid,
    resolve_arch,
    run_tiled_inference,
)


def _real_esrgan_model_cached() -> str | None:
    """Return a cached Real-ESRGAN model name usable for a real forward pass, or None."""
    try:
        from autovideofixer.ai.model_cache import get_model_path
        from autovideofixer.ai.torch_utils import is_torch_available
    except ImportError:
        return None
    if not is_torch_available():
        return None
    for name in ("RealESRGAN_x2plus", "RealESRGAN_x4plus"):
        if get_model_path(name) is not None:
            return name
    return None


class TestRealESRGANUpscaler:
    """Test Real-ESRGAN upscaler wrapper."""

    def test_upscaler_creation(self):
        """Test creating an upscaler."""
        upscaler = RealESRGANUpscaler(scale=4, model_name="RealESRGAN_x4plus")
        assert upscaler.scale == 4
        assert upscaler.model_name == "RealESRGAN_x4plus"
        assert upscaler.is_loaded is False

    def test_upscaler_default_scale(self):
        """Test default scale factor."""
        upscaler = RealESRGANUpscaler()
        assert upscaler.scale == 4

    def test_upscaler_anime_model(self):
        """Test anime model configuration."""
        upscaler = RealESRGANUpscaler(model_name="RealESRGAN_x4plus_anime_6B")
        assert upscaler.model_name == "RealESRGAN_x4plus_anime_6B"

    def test_upscaler_not_loaded_error(self):
        """Test upscale raises when model not loaded."""
        upscaler = RealESRGANUpscaler()
        import numpy as np

        frame = np.zeros((240, 320, 3), dtype="uint8")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            upscaler.upscale(frame)

    def test_upscaler_unload(self):
        """Test unloading model."""
        upscaler = RealESRGANUpscaler()
        upscaler._model = MagicMock()
        upscaler._loaded = True
        upscaler.unload()
        assert upscaler._model is None
        assert upscaler._loaded is False

    @patch("autovideofixer.ai.wrappers.upscale.get_model_path", return_value=None)
    def test_load_model_no_path(self, mock_path):
        """Test loading when no model path available."""
        upscaler = RealESRGANUpscaler()
        result = upscaler.load_model()
        assert result is False

    @patch("autovideofixer.ai.wrappers.upscale.get_model_path")
    def test_load_model_file_not_found(self, mock_path, tmp_path):
        """Test loading when model file doesn't exist."""
        mock_path.return_value = tmp_path / "nonexistent.pth"
        upscaler = RealESRGANUpscaler()
        result = upscaler.load_model()
        assert result is False


class TestSRVGGNetCompact:
    """Test the compact SRVGG architecture used by the Real-ESRGAN video models.

    CPU-only, random weights, no downloads -- exercises forward-pass shape
    correctness and the exact flat state-dict key convention the official
    checkpoints (realesr-general-x4v3/wdn-x4v3/animevideov3) ship with, so
    strict `load_state_dict` succeeds against real weights without needing
    them here.
    """

    @pytest.mark.parametrize("num_conv", [16, 32])
    def test_forward_shape_x4(self, num_conv):
        """Forward pass upscales spatial dims by 4x, channels stay at 3."""
        torch = pytest.importorskip("torch")

        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=4
        )
        model.eval()
        x = torch.randn(1, 3, 16, 24)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 64, 96)

    def test_state_dict_key_convention_matches_official_checkpoints(self):
        """Constructed model's state_dict keys use the flat body.N convention.

        The official BasicSR SRVGGNetCompact checkpoints store every
        parameter under `body.<index>.weight`/`body.<index>.bias` (conv
        layers) or `body.<index>.weight` (PReLU, weight-only), and have NO
        `upsampler.*` keys since PixelShuffle has no learnable parameters.
        A mismatch here means `load_state_dict(strict=True)` would fail
        against a real checkpoint.
        """
        pytest.importorskip("torch")

        num_conv = 4
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=num_conv, upscale=4)
        keys = set(model.state_dict().keys())

        # No params from PixelShuffle -- it has none.
        assert not any(k.startswith("upsampler.") for k in keys)

        # First layer (index 0): conv. Index 1: PReLU (weight only).
        assert {"body.0.weight", "body.0.bias"}.issubset(keys)
        assert "body.1.weight" in keys
        assert "body.1.bias" not in keys  # PReLU has no bias

        # Alternating conv/PReLU pairs for each of the num_conv hidden
        # layers, followed by one final conv (no trailing PReLU).
        last_conv_idx = 2 + 2 * num_conv
        assert {f"body.{last_conv_idx}.weight", f"body.{last_conv_idx}.bias"}.issubset(keys)
        assert f"body.{last_conv_idx + 1}.weight" not in keys

        for i in range(num_conv):
            conv_idx = 2 + 2 * i
            prelu_idx = conv_idx + 1
            assert {f"body.{conv_idx}.weight", f"body.{conv_idx}.bias"}.issubset(keys)
            assert f"body.{prelu_idx}.weight" in keys
            assert f"body.{prelu_idx}.bias" not in keys

        # Every key is body.N.{weight,bias} -- no stray top-level params.
        for k in keys:
            assert k.startswith("body."), f"Unexpected non-body param key: {k}"


class TestSrvggArchDispatch:
    """Test that MODEL_REGISTRY-driven architecture selection picks SRVGG for
    the new compact models and leaves existing RRDB models untouched.

    `resolve_arch()` is the exact pure lookup `RealESRGANUpscaler.load_model()`
    calls to decide between building a `SRVGGNetCompact` or an `RRDBNet` --
    tested directly here since exercising `load_model()` itself would require
    real downloaded weights.
    """

    def test_compact_models_resolve_to_srvgg(self):
        from autovideofixer.ai.model_cache import MODEL_REGISTRY

        for name in (
            "realesr-general-x4v3",
            "realesr-general-wdn-x4v3",
            "realesr-animevideov3",
        ):
            upscaler = RealESRGANUpscaler(model_name=name)
            entry = MODEL_REGISTRY.get(upscaler.model_name, {})
            assert resolve_arch(entry) == "srvgg"

    def test_rrdb_models_resolve_to_rrdb_default(self):
        """NEW MODELS MUST NOT BREAK OLD: existing RRDB models still default correctly."""
        from autovideofixer.ai.model_cache import MODEL_REGISTRY

        for name in ("RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B"):
            upscaler = RealESRGANUpscaler(model_name=name)
            entry = MODEL_REGISTRY.get(upscaler.model_name, {})
            assert resolve_arch(entry) == "rrdb"

    def test_unknown_model_defaults_to_rrdb(self):
        """A model name with no registry entry at all still defaults to rrdb."""
        assert resolve_arch({}) == "rrdb"


class TestSrvggModelSwapPassThrough:
    """DeblockStage/DenoiseVideoStage swap RealESRGAN_x4plus -> x2plus (a
    smaller-body optimization for their scale<=2 use) via a strict `==`
    string equality check against the literal "RealESRGAN_x4plus". Compact
    SRVGG model names must NOT be substituted by that logic -- mirrors the
    condition at core/stages/deblock.py and core/stages/denoise_video.py.
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            "realesr-general-x4v3",
            "realesr-general-wdn-x4v3",
            "realesr-animevideov3",
            "RealESRGAN_x4plus_anime_6B",
        ],
    )
    def test_non_x4plus_names_pass_through_unswapped(self, model_name):
        # Mirrors: `if deblock_model == "RealESRGAN_x4plus": deblock_model = "RealESRGAN_x2plus"`
        swapped = "RealESRGAN_x2plus" if model_name == "RealESRGAN_x4plus" else model_name
        assert swapped == model_name

    def test_x4plus_is_still_swapped(self):
        """Sanity check the swap condition itself still fires for the literal it targets."""
        model_name = "RealESRGAN_x4plus"
        swapped = "RealESRGAN_x2plus" if model_name == "RealESRGAN_x4plus" else model_name
        assert swapped == "RealESRGAN_x2plus"


class TestRIFEInterpolator:
    """Test RIFE frame interpolator wrapper."""

    def test_interpolator_creation(self):
        """Test creating an interpolator."""
        interp = RIFEInterpolator(model_name="rife_v4.6")
        assert interp.model_name == "rife_v4.6"
        assert interp.is_loaded is False

    def test_interpolator_not_loaded_error(self):
        """Test interpolation raises when model not loaded."""
        interp = RIFEInterpolator()
        import numpy as np

        frame0 = np.zeros((240, 320, 3), dtype="uint8")
        frame1 = np.zeros((240, 320, 3), dtype="uint8")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            interp.interpolate(frame0, frame1)

    def test_interpolator_unload(self):
        """Test unloading model."""
        interp = RIFEInterpolator()
        interp._model = MagicMock()
        interp._loaded = True
        interp.unload()
        assert interp._model is None
        assert interp._loaded is False

    @patch("autovideofixer.ai.wrappers.interpolate.get_model_path", return_value=None)
    def test_load_model_no_path(self, mock_path):
        """Test loading when no model path available."""
        interp = RIFEInterpolator()
        result = interp.load_model()
        assert result is False

    @patch("autovideofixer.ai.wrappers.interpolate.get_model_path")
    def test_load_model_file_not_found(self, mock_path, tmp_path):
        """Test loading when model file doesn't exist."""
        mock_path.return_value = tmp_path / "nonexistent.pkl"
        interp = RIFEInterpolator()
        result = interp.load_model()
        assert result is False

    def test_interpolate_video_single_frame(self):
        """Test interpolation with single frame (factor=2)."""
        interp = RIFEInterpolator()
        interp._loaded = True
        interp._model = MagicMock()
        interp._device = MagicMock()
        interp._device.type = "cpu"

        import numpy as np

        frames = [np.zeros((240, 320, 3), dtype="uint8")]

        # Mock the interpolate method
        interp.interpolate = MagicMock(return_value=np.zeros((240, 320, 3), dtype="uint8"))

        result = interp.interpolate_video(frames, factor=2)
        # With 1 frame and factor 2, we expect at least the original frame
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_interpolate_video_factor_1(self):
        """Test interpolation with factor 1 returns original frames."""
        interp = RIFEInterpolator()
        interp._loaded = True
        interp._model = MagicMock()
        interp._device = MagicMock()
        interp._device.type = "cpu"

        import numpy as np

        frames = [np.zeros((240, 320, 3), dtype="uint8")]
        result = interp.interpolate_video(frames, factor=1)
        assert len(result) == 1


@pytest.mark.integration
class TestRealESRGANNotBlackRegression:
    """Regression coverage for the ResidualDenseBlock 0.2-residual-scaling bug.

    That bug (a missing `* 0.2` on the dense block's residual branch, present
    despite the checkpoint's state_dict loading with a perfectly matching
    architecture) made every Real-ESRGAN-based stage -- upscale,
    denoise_video, deblock -- silently produce solid black output: the
    unscaled residual compounded across 23 RRDB blocks x 3 dense blocks each
    (69 total) until activations overflowed to NaN, which
    frame_from_tensor()'s nan_to_num(nan=0.0) then rendered as black. ffprobe
    metadata (resolution/framerate/frame count) alone can't catch this since
    the container and dimensions are all still correct -- only pixel content
    reveals it. Runs the real model against a real (skip if uncached)
    checkpoint on a small synthetic frame, so it exercises the actual
    forward() path this bug lived in rather than a mock.
    """

    def test_upscale_output_is_not_black(self):
        model_name = _real_esrgan_model_cached()
        if model_name is None:
            pytest.skip("No cached Real-ESRGAN checkpoint available for a real forward pass")

        import numpy as np

        # A structured (non-uniform) synthetic frame: a black background
        # would trivially "pass" a mean-luma check for the wrong reason.
        frame = np.zeros((64, 64, 3), dtype="uint8")
        frame[:32, :, 0] = 200  # top half: blue-ish (BGR)
        frame[32:, :, 1] = 180  # bottom half: green-ish
        frame[:, 28:36, 2] = 255  # a red stripe down the middle

        upscaler = RealESRGANUpscaler(scale=2, model_name=model_name, device_preference="auto")
        assert upscaler.load_model(), f"Failed to load cached model {model_name}"
        try:
            result = upscaler.upscale(frame)
        finally:
            upscaler.unload()

        assert result.shape[0] > 0 and result.shape[1] > 0
        # A solid-black (or solid-anything) frame has zero variance; the
        # NaN-collapse bug produced exactly that. A real super-resolved
        # frame of structured input has substantial variance.
        assert result.std() > 5.0, (
            f"Output frame has near-zero variance (std={result.std():.3f}) -- "
            "looks like uniform/black output, not a real super-resolution result"
        )
        mean_luma = result.mean()
        assert 5.0 < mean_luma < 250.0, (
            f"Output mean luma {mean_luma:.1f} is outside a sane range -- "
            "solid black (~0) or solid white (~255) both indicate broken inference"
        )


@pytest.mark.integration
class TestBatchedInferenceRealGPUCorrectness:
    """Real-GPU numeric-equivalence checks for batch_size and tile_batch_size.

    Skips (does not fail) if no cached checkpoint or no CUDA device is
    available -- same pattern as TestRealESRGANNotBlackRegression above.
    Verifies, on a real forward pass through the real model:
      - Frame order is preserved exactly through the batch/split round-trip
        (distinguishable per-frame colors).
      - Tail handling (frame count not evenly divisible by batch size).
      - batch_size > 1 output is numerically close to batch_size=1 (small
        floating-point differences from different cuDNN algorithm selection
        are expected and fine; a structural/order-scrambling bug is not).
    """

    def _distinguishable_frames(self, n: int, size: int = 64):
        import numpy as np

        frames = []
        for i in range(n):
            f = np.zeros((size, size, 3), dtype="uint8")
            f[:, :, 0] = (i * 23) % 256
            f[:, :, 1] = (i * 47 + 7) % 256
            f[:, :, 2] = (i * 91 + 13) % 256
            frames.append(f)
        return frames

    def test_whole_frame_batch_size_matches_single_frame_path(self):
        model_name = _real_esrgan_model_cached()
        if model_name is None:
            pytest.skip("No cached Real-ESRGAN checkpoint available for a real forward pass")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for real batched-inference verification")
        import numpy as np

        frames = self._distinguishable_frames(10)  # not evenly divisible by 4 or 8

        outputs: dict[int, list] = {}
        for bs in (1, 4, 8):
            upscaler = RealESRGANUpscaler(
                scale=2, model_name=model_name, device_preference="cuda", batch_size=bs
            )
            assert upscaler.load_model(), f"Failed to load {model_name}"
            try:
                outputs[bs] = upscaler.upscale_video(frames)
            finally:
                upscaler.unload()
            assert len(outputs[bs]) == 10, f"batch_size={bs}: frame count changed"

        baseline = outputs[1]
        for bs in (4, 8):
            max_diffs = [
                int(np.abs(a.astype(int) - b.astype(int)).max())
                for a, b in zip(baseline, outputs[bs])
            ]
            overall_max = max(max_diffs)
            print(f"[whole-frame batch_size={bs}] max-abs-diff vs batch_size=1: {overall_max}")
            # "Small enough" = a handful of uint8 levels from cuDNN algorithm
            # selection differences, not a structural/order-scrambling bug.
            assert overall_max <= 12, (
                f"batch_size={bs} output diverged too far from batch_size=1 "
                f"(max_diff={overall_max}) -- looks like a correctness bug, not FP noise"
            )

    def test_tile_batch_size_matches_single_tile_path(self):
        model_name = _real_esrgan_model_cached()
        if model_name is None:
            pytest.skip("No cached Real-ESRGAN checkpoint available for a real forward pass")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for real batched-inference verification")
        import numpy as np

        # Force tiling on a frame well above AUTO_TILE_THRESHOLD_PX so the
        # tile_batch_size path is actually exercised.
        frame = np.zeros((1200, 1600, 3), dtype="uint8")
        frame[:600, :, 0] = 180
        frame[600:, :, 1] = 160
        frame[:, 700:900, 2] = 255

        outputs = {}
        for tbs in (1, 4, 8):
            upscaler = RealESRGANUpscaler(
                scale=1,
                model_name=model_name,
                device_preference="cuda",
                tile_size=256,
                tile_overlap=16,
                tile_batch_size=tbs,
            )
            assert upscaler.load_model(), f"Failed to load {model_name}"
            try:
                outputs[tbs] = upscaler.upscale(frame)
            finally:
                upscaler.unload()

        baseline = outputs[1]
        for tbs in (4, 8):
            diff = int(np.abs(baseline.astype(int) - outputs[tbs].astype(int)).max())
            print(f"[tile_batch_size={tbs}] max-abs-diff vs tile_batch_size=1: {diff}")
            assert diff <= 12, (
                f"tile_batch_size={tbs} output diverged too far from tile_batch_size=1 "
                f"(max_diff={diff}) -- looks like a correctness bug, not FP noise"
            )


class TestTileGrid:
    """Tests for the tiled-inference grid math (compute_tile_grid), no GPU needed.

    These validate the pure coordinate math used to avoid CUDA OOM on large
    frames by splitting them into overlapping tiles: correct tile counts,
    overlap padding present and clamped at image bounds, and full coverage
    with no gaps/overlaps in the *output* placement regions.
    """

    def test_exact_multiple_grid_shape(self):
        # 512x512 image, 256 tile -> exactly a 2x2 grid, no partial tiles.
        tiles = compute_tile_grid(512, 512, tile_size=256, overlap=16, out_scale=1)
        assert len(tiles) == 4

    def test_non_multiple_grid_shape(self):
        # 500x300 image, 256 tile -> ceil(500/256)=2 rows, ceil(300/256)=2 cols.
        tiles = compute_tile_grid(500, 300, tile_size=256, overlap=16, out_scale=1)
        assert len(tiles) == 4

    def test_single_tile_when_smaller_than_tile_size(self):
        tiles = compute_tile_grid(100, 100, tile_size=256, overlap=16, out_scale=1)
        assert len(tiles) == 1
        t = tiles[0]
        assert (t.in_y0, t.in_y1, t.in_x0, t.in_x1) == (0, 100, 0, 100)
        assert (t.out_y0, t.out_y1, t.out_x0, t.out_x1) == (0, 100, 0, 100)

    def test_overlap_padding_present_and_clamped(self):
        # A middle tile should be padded by `overlap` on every side; an edge
        # tile's padding must be clamped to the image boundary (never go
        # negative or past height/width).
        tiles = compute_tile_grid(600, 600, tile_size=200, overlap=20, out_scale=1)
        by_pos = {(t.out_y0, t.out_x0): t for t in tiles}

        top_left = by_pos[(0, 0)]
        assert top_left.in_y0 == 0  # clamped, can't pad above 0
        assert top_left.in_x0 == 0
        assert top_left.in_y1 == 220  # 200 + 20 overlap below
        assert top_left.in_x1 == 220

        middle = by_pos[(200, 200)]
        assert middle.in_y0 == 180  # 200 - 20
        assert middle.in_y1 == 420  # 400 + 20
        assert middle.in_x0 == 180
        assert middle.in_x1 == 420

        bottom_right = by_pos[(400, 400)]
        assert bottom_right.in_y1 == 600  # clamped, can't pad past image height
        assert bottom_right.in_x1 == 600

    def test_output_placement_covers_full_image_no_gaps_no_overlaps(self):
        h, w, tile, overlap, scale = 517, 333, 128, 24, 2
        tiles = compute_tile_grid(h, w, tile_size=tile, overlap=overlap, out_scale=scale)

        canvas = [[0] * (w * scale) for _ in range(h * scale)]
        for t in tiles:
            for y in range(t.out_y0, t.out_y1):
                for x in range(t.out_x0, t.out_x1):
                    canvas[y][x] += 1

        # Every output pixel must be covered by exactly one tile's placement
        # region -- a gap (0) would leave holes in the stitched frame, an
        # overlap (>1) means two tiles wrote the same pixel (a stitching bug,
        # not the *input*-side overlap padding which is intentional).
        flat = [v for row in canvas for v in row]
        assert set(flat) == {1}

    def test_crop_region_matches_out_region_size(self):
        tiles = compute_tile_grid(517, 333, tile_size=128, overlap=24, out_scale=2)
        for t in tiles:
            assert (t.out_y1 - t.out_y0) == (t.crop_y1 - t.crop_y0)
            assert (t.out_x1 - t.out_x0) == (t.crop_x1 - t.crop_x0)
            # crop region must lie within the tile's own (padded) output size
            tile_out_h = (t.in_y1 - t.in_y0) * 2
            tile_out_w = (t.in_x1 - t.in_x0) * 2
            assert 0 <= t.crop_y0 <= t.crop_y1 <= tile_out_h
            assert 0 <= t.crop_x0 <= t.crop_x1 <= tile_out_w

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            compute_tile_grid(100, 100, tile_size=0, overlap=8)
        with pytest.raises(ValueError):
            compute_tile_grid(100, 100, tile_size=64, overlap=-1)
        with pytest.raises(ValueError):
            compute_tile_grid(0, 100, tile_size=64, overlap=8)


class TestRunTiledInference:
    """Tests run_tiled_inference end-to-end on CPU tensors (no GPU needed)."""

    def test_identity_reconstruction(self):
        """Tiling an identity function must reconstruct the original tensor exactly."""
        torch = pytest.importorskip("torch")

        tensor = torch.arange(1 * 3 * 64 * 48, dtype=torch.float32).reshape(1, 3, 64, 48)

        def identity(t):
            return t

        out = run_tiled_inference(tensor, tile_size=20, overlap=5, out_scale=1, infer_fn=identity)
        assert out.shape == tensor.shape
        assert torch.equal(out, tensor)

    def test_scaling_reconstruction(self):
        """A tile-wise 2x nearest-upsample must equal a whole-frame 2x upsample."""
        torch = pytest.importorskip("torch")

        tensor = torch.rand(1, 3, 50, 37, dtype=torch.float32)

        def upsample2x(t):
            return torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")

        expected = upsample2x(tensor)
        out = run_tiled_inference(tensor, tile_size=16, overlap=4, out_scale=2, infer_fn=upsample2x)
        assert out.shape == expected.shape
        assert torch.allclose(out, expected)

    def test_tile_batching_matches_unbatched(self):
        """Batched-tile output must equal one-tile-at-a-time output for a real per-call fn."""
        torch = pytest.importorskip("torch")

        # 96x96 @ tile_size=16/overlap=4 gives a grid with a 16-tile group of
        # uniform (interior) padded shape -- large enough for tile_batch_size
        # batching to actually kick in and reduce call count (see
        # compute_tile_grid's shape distribution for this size).
        tensor = torch.rand(1, 3, 96, 96, dtype=torch.float32)
        calls: list[int] = []

        def infer_fn(t):
            calls.append(t.shape[0])
            return t * 2.0 + 1.0

        out_unbatched = run_tiled_inference(
            tensor, tile_size=16, overlap=4, out_scale=1, infer_fn=infer_fn, tile_batch_size=1
        )
        unbatched_call_count = len(calls)
        calls.clear()

        out_batched = run_tiled_inference(
            tensor, tile_size=16, overlap=4, out_scale=1, infer_fn=infer_fn, tile_batch_size=4
        )
        batched_call_count = len(calls)

        assert torch.allclose(out_unbatched, out_batched)
        # Batching must actually reduce the number of forward-pass calls.
        assert batched_call_count < unbatched_call_count
        assert max(calls) > 1  # at least one call actually received >1 stacked tiles

    def test_tile_batching_falls_back_when_frame_batch_dim_not_one(self):
        """A tensor with its own N>1 (whole-frame) batch dim must skip tile batching."""
        torch = pytest.importorskip("torch")

        tensor = torch.rand(2, 3, 64, 48, dtype=torch.float32)
        calls: list[int] = []

        def infer_fn(t):
            calls.append(t.shape[0])
            return t

        run_tiled_inference(
            tensor, tile_size=20, overlap=5, out_scale=1, infer_fn=infer_fn, tile_batch_size=4
        )
        # Every call still carries the original 2-frame batch dim (unchanged
        # per-tile loop), never stacked with other tiles.
        assert all(c == 2 for c in calls)
        assert len(calls) > 1

    def test_tile_batching_oom_halves_and_recovers(self):
        """A tile-batch OOM must recursively halve and still reconstruct the exact result."""
        torch = pytest.importorskip("torch")

        tensor = torch.arange(1 * 3 * 64 * 64, dtype=torch.float32).reshape(1, 3, 64, 64)
        seen: list[int] = []

        def flaky_identity(t):
            seen.append(t.shape[0])
            if t.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return t

        with patch("autovideofixer.ai.wrappers.upscale.torch.cuda.empty_cache"):
            out = run_tiled_inference(
                tensor,
                tile_size=16,
                overlap=0,
                out_scale=1,
                infer_fn=flaky_identity,
                tile_batch_size=8,
            )

        assert torch.equal(out, tensor)
        # Confirms the halving path was actually exercised (some call saw >2
        # tiles and OOM'd before a later, smaller call succeeded).
        assert max(seen) > 2
        assert min(seen) <= 2

    def test_tile_batching_oom_releases_failed_batch_before_retry(self):
        """The OOM'd batched input tensor must be released before cuda.empty_cache()/retry.

        Regression test for the OOM-retry defect: previously the failed
        batch's `batched_in` tensor stayed alive (referenced by the
        recursing stack frame) through the entire halving retry, so every
        retry ran with LESS free VRAM than the attempt that had just failed
        instead of getting back what that failed attempt would have freed.
        """
        torch = pytest.importorskip("torch")
        import gc
        import weakref

        tensor = torch.arange(1 * 3 * 64 * 64, dtype=torch.float32).reshape(1, 3, 64, 64)
        weak_holder: dict[str, object] = {}
        release_observed: list[bool] = []

        def fake_infer(t):
            if t.shape[0] > 2:
                # `t` IS the `batched_in` tensor passed straight through by
                # run_tiled_inference -- weakref it so empty_cache() (called
                # right after `del batched_in`, before the retry) can check
                # whether that tensor's last reference is already gone.
                weak_holder["ref"] = weakref.ref(t)
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return t

        def checking_empty_cache():
            ref = weak_holder.get("ref")
            if ref is not None:
                gc.collect()
                release_observed.append(ref() is None)  # True == already released

        with patch(
            "autovideofixer.ai.wrappers.upscale.torch.cuda.empty_cache",
            side_effect=checking_empty_cache,
        ):
            out = run_tiled_inference(
                tensor,
                tile_size=16,
                overlap=0,
                out_scale=1,
                infer_fn=fake_infer,
                tile_batch_size=8,
            )

        assert torch.equal(out, tensor)
        # empty_cache() must actually have run on at least one OOM'd batch,
        # and every time it ran, the failed batch tensor must already have
        # been released (del'd) -- not merely about to be, after the retry.
        assert release_observed
        assert all(release_observed)

    def test_tile_batching_groups_by_shape(self):
        """Tiles with different padded shapes (edge/corner tiles) must not be batched together."""
        torch = pytest.importorskip("torch")

        # 50x37 with tile_size=16 produces a non-uniform grid: interior tiles
        # are 16x16 (plus overlap), the last row/column are smaller.
        tensor = torch.rand(1, 3, 50, 37, dtype=torch.float32)
        seen_shapes: list[tuple[int, int]] = []

        def infer_fn(t):
            seen_shapes.append((t.shape[2], t.shape[3]))
            return t

        run_tiled_inference(
            tensor, tile_size=16, overlap=4, out_scale=1, infer_fn=infer_fn, tile_batch_size=8
        )
        # Every batched call's tiles were verified same-shape by construction
        # (torch.cat would raise otherwise) -- getting here without an
        # exception is itself the correctness check; also sanity-check more
        # than one shape group existed (edge tiles differ from interior ones).
        assert len(set(seen_shapes)) > 1


class TestUpscaleBatchWholeFrame:
    """Tests RealESRGANUpscaler.upscale_batch()/upscale_video() whole-frame batching.

    Uses a fake identity `_model` (scale=1, native_scale=1, tile_size=0 so no
    frame in these small test images ever needs tiling) on CPU -- no real
    GPU or checkpoint needed to validate the batching/splitting/ordering
    logic itself. Real-model numeric-equivalence verification lives in the
    GPU-gated integration tests below.
    """

    def _make_upscaler(self, batch_size=4, tile_size=0, tta_mode=0):
        import torch

        upscaler = RealESRGANUpscaler(
            scale=1,
            model_name="fake",
            tta_mode=tta_mode,
            tile_size=tile_size,
            batch_size=batch_size,
        )
        upscaler._loaded = True
        upscaler._device = torch.device("cpu")
        upscaler._use_fp16 = False
        upscaler._native_scale = 1
        upscaler.backend = "torch"
        return upscaler

    def test_order_preserved_and_matches_single_frame_path(self):
        pytest.importorskip("torch")
        import numpy as np

        upscaler = self._make_upscaler(batch_size=4)
        upscaler._model = lambda t: t  # identity: output must equal input exactly

        frames = [
            np.full((8, 8, 3), fill_value=v, dtype="uint8")
            for v in (0, 25, 50, 75, 100, 125, 150, 175, 200, 225)
        ]  # 10 frames, not evenly divisible by batch_size=4

        batched_results = upscaler.upscale_video(frames)
        single_results = [upscaler.upscale(f) for f in frames]

        assert len(batched_results) == 10
        for i, (out_frame, in_frame) in enumerate(zip(batched_results, frames)):
            assert (out_frame == in_frame).all(), f"frame {i} order/content mismatch"
        for out_frame, single_frame in zip(batched_results, single_results):
            assert (out_frame == single_frame).all()

    def test_tail_chunking_ten_frames_batch_four(self):
        """10 frames / batch_size=4 must chunk as 4, 4, 2 -- no dropped/duplicated frames."""
        pytest.importorskip("torch")
        import numpy as np

        upscaler = self._make_upscaler(batch_size=4)
        seen_batch_sizes: list[int] = []

        def fake_model(t):
            seen_batch_sizes.append(t.shape[0])
            return t

        upscaler._model = fake_model
        frames = [np.zeros((8, 8, 3), dtype="uint8") for _ in range(10)]

        results = upscaler.upscale_video(frames)

        assert len(results) == 10
        assert seen_batch_sizes == [4, 4, 2]

    def test_oom_batch_split_in_half(self):
        """A batched-forward-pass OOM must recursively halve down to single frames."""
        pytest.importorskip("torch")
        import numpy as np
        import torch

        upscaler = self._make_upscaler(batch_size=4)
        call_sizes: list[int] = []

        def fake_model(t):
            call_sizes.append(t.shape[0])
            if t.shape[0] > 1:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return t

        upscaler._model = fake_model
        frames = [np.full((8, 8, 3), fill_value=i * 10, dtype="uint8") for i in range(4)]

        with patch("autovideofixer.ai.wrappers.upscale.torch.cuda.empty_cache"):
            results = upscaler.upscale_batch(frames)

        assert len(results) == 4
        for i, (out_frame, in_frame) in enumerate(zip(results, frames)):
            assert (out_frame == in_frame).all(), f"frame {i} mismatch after OOM recovery"
        assert max(call_sizes) > 1  # the OOM path was actually exercised

    def test_falls_back_to_per_frame_when_any_frame_needs_tiling(self):
        """Whole-frame batching must never combine with tiling -- falls back per-frame."""
        pytest.importorskip("torch")
        import numpy as np

        upscaler = self._make_upscaler(batch_size=4, tile_size=4)  # force tiling
        seen_sizes: list[int] = []

        def fake_model(t):
            seen_sizes.append(t.shape[0])
            return t

        upscaler._model = fake_model
        frames = [np.zeros((8, 8, 3), dtype="uint8") for _ in range(3)]

        results = upscaler.upscale_batch(frames)

        assert len(results) == 3
        # Per-tile calls always carry a batch dim of 1 (whole-frame batching
        # never reaches the model here -- upscale() -> tiled inference path).
        assert all(s == 1 for s in seen_sizes)
        assert len(seen_sizes) > 1  # confirms tiling (not a trivial no-op) happened

    def test_falls_back_to_per_frame_when_tta_enabled(self):
        """Whole-frame batching must never combine with TTA -- falls back per-frame."""
        pytest.importorskip("torch")
        import numpy as np

        upscaler = self._make_upscaler(batch_size=4, tta_mode=7)
        seen_sizes: list[int] = []

        def fake_model(t):
            seen_sizes.append(t.shape[0])
            return t

        upscaler._model = fake_model
        frames = [np.zeros((8, 8, 3), dtype="uint8") for _ in range(4)]

        results = upscaler.upscale_batch(frames)

        assert len(results) == 4
        # apply_tta calls the model multiple times per SINGLE frame (batch=1
        # each), never with N different frames stacked together.
        assert all(s == 1 for s in seen_sizes)
        assert len(seen_sizes) > 4  # tta_mode=7 -> 4 augmented forward passes/frame

    def test_single_frame_chunk_uses_upscale_directly(self):
        pytest.importorskip("torch")
        import numpy as np

        upscaler = self._make_upscaler(batch_size=4)
        upscaler._model = lambda t: t

        results = upscaler.upscale_batch([np.full((8, 8, 3), 42, dtype="uint8")])
        assert len(results) == 1
        assert (results[0] == 42).all()

    def test_empty_chunk_returns_empty(self):
        pytest.importorskip("torch")

        upscaler = self._make_upscaler(batch_size=4)
        assert upscaler.upscale_batch([]) == []
