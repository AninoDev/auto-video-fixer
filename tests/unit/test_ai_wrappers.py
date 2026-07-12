"""Tests for AI model wrappers (Real-ESRGAN, RIFE)."""

from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.ai.wrappers.interpolate import RIFEInterpolator
from autovideofixer.ai.wrappers.upscale import (
    RealESRGANUpscaler,
    compute_tile_grid,
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
