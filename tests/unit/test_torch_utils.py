"""Tests for AI torch utilities."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.ai.torch_utils import (
    frame_from_tensor,
    get_device,
    get_dtype,
    is_torch_available,
    tensor_from_frame,
)


class TestIsTorchAvailable:
    """Test PyTorch availability detection."""

    def test_torch_available(self):
        """Test detection when PyTorch is installed."""
        # PyTorch is in the optional deps; if installed, should return True
        result = is_torch_available()
        # Just verify it doesn't raise
        assert isinstance(result, bool)

    def test_torch_not_available(self):
        """Test detection when PyTorch is not installed."""
        with patch.dict(sys.modules, {"torch": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = is_torch_available()
                assert result is False


class TestGetDevice:
    """Test device detection."""

    def test_get_device_auto(self):
        """Test get_device auto-detection."""
        device = get_device("auto")
        assert device is not None

    def test_get_dtype(self):
        """Test dtype selection."""
        dtype_fp32 = get_dtype("fp32")
        assert dtype_fp32 is not None

        dtype_fp16 = get_dtype("fp16")
        assert dtype_fp16 is not None

    def test_get_device_cpu(self):
        """Test get_device CPU."""
        device = get_device("cpu")
        assert str(device) == "cpu"

    def test_tensor_from_frame_no_torch(self):
        """Test tensor_from_frame with mocked torch."""
        import numpy as np

        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value = MagicMock()
        mock_torch.float32 = "float32"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch("autovideofixer.ai.torch_utils.is_torch_available", return_value=True):
                frame = np.zeros((240, 320, 3), dtype="uint8")
                result = tensor_from_frame(frame)
                assert result is not None

    def test_frame_from_tensor_no_torch(self):
        """Test frame_from_tensor with mocked torch."""
        mock_torch = MagicMock()
        mock_torch.nn.functional.interpolate.return_value = MagicMock()
        mock_torch.detach = MagicMock()

        mock_arr = MagicMock()
        mock_arr.squeeze.return_value = MagicMock()
        mock_arr.transpose.return_value = MagicMock()
        mock_arr.numpy.return_value = MagicMock()
        mock_arr.clip.return_value = MagicMock()
        mock_torch.from_numpy = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch("autovideofixer.ai.torch_utils.is_torch_available", return_value=True):
                tensor = MagicMock()
                tensor.detach.return_value.cpu.return_value = MagicMock()
                tensor.squeeze.return_value.numpy.return_value.transpose.return_value = MagicMock()
                result = frame_from_tensor(tensor)
                assert result is not None


class TestBatchedTensorConversion:
    """Tests for tensor_from_frames/frames_from_tensor (real CPU torch, no GPU needed).

    These back the whole-frame batching path in
    ai/wrappers/upscale.py's RealESRGANUpscaler.upscale_batch() -- the
    single-frame tensor_from_frame/frame_from_tensor functions above are
    unchanged and still used by ncnn backends and RIFE interpolation.
    """

    def test_shape_is_batched_not_unsqueezed(self):
        torch = pytest.importorskip("torch")
        import numpy as np

        from autovideofixer.ai.torch_utils import tensor_from_frames

        frames = [np.zeros((4, 6, 3), dtype="uint8") for _ in range(3)]
        t = tensor_from_frames(frames, device=torch.device("cpu"))
        assert t.shape == (3, 3, 4, 6)

    def test_batched_conversion_matches_per_frame_conversion(self):
        """Order-preserving equivalence: tensor_from_frames == cat of tensor_from_frame calls."""
        torch = pytest.importorskip("torch")
        import numpy as np

        from autovideofixer.ai.torch_utils import tensor_from_frames

        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (8, 8, 3), dtype="uint8") for _ in range(4)]

        batched = tensor_from_frames(frames, device=torch.device("cpu"))
        singles = torch.cat(
            [tensor_from_frame(f, device=torch.device("cpu")) for f in frames], dim=0
        )
        assert torch.equal(batched, singles)

    def test_frames_from_tensor_roundtrip_matches_single_frame_path(self):
        """frames_from_tensor split output must equal per-frame frame_from_tensor output."""
        torch = pytest.importorskip("torch")
        import numpy as np

        from autovideofixer.ai.torch_utils import frames_from_tensor, tensor_from_frames

        rng = np.random.default_rng(1)
        frames = [rng.integers(0, 256, (8, 8, 3), dtype="uint8") for _ in range(4)]

        batched_in = tensor_from_frames(frames, device=torch.device("cpu"))
        out_frames = frames_from_tensor(batched_in, scale=1.0)

        assert len(out_frames) == 4
        for i, (out_frame, orig) in enumerate(zip(out_frames, frames)):
            single = frame_from_tensor(batched_in[i : i + 1], scale=1.0)
            assert (out_frame == single).all(), f"frame {i} split mismatch"
            # tensor_from_frames -> frames_from_tensor with an identity in
            # between (no model) must round-trip exactly for uint8 input.
            assert (out_frame == orig).all(), f"frame {i} order/content mismatch"

    def test_order_preserved_with_distinguishable_frames(self):
        """Output index i must correspond to input frames[i] through the batch round-trip."""
        torch = pytest.importorskip("torch")
        import numpy as np

        from autovideofixer.ai.torch_utils import frames_from_tensor, tensor_from_frames

        frames = [np.full((4, 4, 3), fill_value=v, dtype="uint8") for v in (10, 90, 200)]
        batched = tensor_from_frames(frames, device=torch.device("cpu"))
        out = frames_from_tensor(batched, scale=1.0)
        for out_frame, v in zip(out, (10, 90, 200)):
            assert (out_frame == v).all()

    def test_tail_batch_size_not_evenly_divisible(self):
        """A batch of a single leftover frame must still stack/split correctly."""
        torch = pytest.importorskip("torch")
        import numpy as np

        from autovideofixer.ai.torch_utils import frames_from_tensor, tensor_from_frames

        frames = [np.full((4, 4, 3), fill_value=7, dtype="uint8")]
        batched = tensor_from_frames(frames, device=torch.device("cpu"))
        assert batched.shape[0] == 1
        out = frames_from_tensor(batched, scale=1.0)
        assert len(out) == 1
        assert (out[0] == 7).all()


class TestPinnedStagingPoolKeying:
    """Mocked tests for PinnedStagingPool's key/round-robin/eviction logic.

    Patches `torch.empty` so these run without a real CUDA device (no actual
    page-locking happens) -- purely exercising the pool's bookkeeping, not
    the H2D-copy race the pool exists to prevent (see
    TestPinnedStagingPoolCudaCorrectness below for that, which needs a real
    CUDA stream).
    """

    def _fake_empty_factory(self, torch):
        created = []
        real_empty = torch.empty  # capture BEFORE patching, or fake_empty would recurse into itself

        def fake_empty(shape, dtype=None, pin_memory=False):
            t = real_empty(shape, dtype=dtype)  # real tensor, just not pinned
            created.append(t)
            return t

        return fake_empty, created

    def test_round_robins_slots_for_same_key(self):
        """Repeated get_slot() calls for one key must cycle through SLOTS_PER_KEY tensors."""
        torch = pytest.importorskip("torch")
        from autovideofixer.ai.torch_utils import PinnedStagingPool

        fake_empty, created = self._fake_empty_factory(torch)
        pool = PinnedStagingPool()
        with patch("torch.empty", side_effect=fake_empty):
            slots = [pool.get_slot((1, 4, 4, 3), torch.uint8) for _ in range(5)]

        assert len(created) == PinnedStagingPool.SLOTS_PER_KEY  # allocated once per key, reused
        # Round-robin: slot N and slot N + SLOTS_PER_KEY must be the same object.
        assert slots[0] is slots[PinnedStagingPool.SLOTS_PER_KEY]
        assert slots[0] is not slots[1]

    def test_different_shapes_get_different_keys(self):
        """Different (shape, dtype) pairs must not share pool slots."""
        torch = pytest.importorskip("torch")
        from autovideofixer.ai.torch_utils import PinnedStagingPool

        fake_empty, created = self._fake_empty_factory(torch)
        pool = PinnedStagingPool()
        with patch("torch.empty", side_effect=fake_empty):
            slot_a = pool.get_slot((1, 4, 4, 3), torch.uint8)
            slot_b = pool.get_slot((1, 8, 8, 3), torch.uint8)

        assert slot_a.tensor.shape != slot_b.tensor.shape
        assert len(created) == 2 * PinnedStagingPool.SLOTS_PER_KEY

    def test_evicts_least_recently_used_key_at_cap(self):
        """A new key beyond MAX_KEYS must evict the least-recently-used existing key."""
        torch = pytest.importorskip("torch")
        from autovideofixer.ai.torch_utils import PinnedStagingPool

        fake_empty, created = self._fake_empty_factory(torch)
        pool = PinnedStagingPool()
        assert PinnedStagingPool.MAX_KEYS == 2, "test assumes the documented cap of 2 keys"
        with patch("torch.empty", side_effect=fake_empty):
            pool.get_slot((1,), torch.uint8)  # key A
            pool.get_slot((2,), torch.uint8)  # key B (at cap: A, B)
            pool.get_slot((3,), torch.uint8)  # key C -> evicts A (LRU)
            n_before_readd = len(created)
            pool.get_slot((1,), torch.uint8)  # key A again -> must re-allocate (was evicted)

        assert len(created) > n_before_readd


class TestPinnedStagingPoolCudaCorrectness:
    """Real-CUDA correctness test for the pinned staging pool's slot-reuse ordering.

    This is the test that actually matters for Phase 1.3's correctness
    caveat: a pooled pinned tensor is REUSED across calls, and if a later
    call's copy_() overwrote a slot before its previous non_blocking H2D
    copy had genuinely finished reading from it, the device would receive a
    torn/wrong frame -- silently, nondeterministically. Runs a sequence of
    DIFFERENT frames through the pool (forcing real slot reuse) and asserts
    every one round-trips bit-exact back through frame_from_tensor,
    compared against the same frames run through tensor_from_frame's
    fresh-.pin_memory() (non-pooled) code path taken when device is CPU.
    """

    @pytest.mark.integration
    def test_pooled_h2d_matches_unpooled_across_reused_slots(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        import numpy as np

        from autovideofixer.ai.torch_utils import frame_from_tensor, tensor_from_frame

        device = torch.device("cuda")
        rng = np.random.default_rng(42)
        # More frames than SLOTS_PER_KEY so slots are genuinely reused
        # multiple times within this loop, not just allocated once.
        frames = [rng.integers(0, 256, (16, 16, 3), dtype="uint8") for _ in range(12)]

        # Pooled path (tensor_from_frame's real CUDA branch, which now uses
        # the pool internally).
        pooled_results = []
        for f in frames:
            t = tensor_from_frame(f, device=device)
            pooled_results.append(frame_from_tensor(t, scale=1.0))

        # Reference: convert each frame to a tensor via a fresh (non-pooled)
        # pin_memory() + H2D copy, forcing a full device sync after each one
        # so there's no possible race to compare against.
        reference_results = []
        for f in frames:
            arr = np.ascontiguousarray(f)
            cpu_t = torch.from_numpy(arr).unsqueeze(0).pin_memory()
            t = cpu_t.to(device=device, non_blocking=True)
            torch.cuda.synchronize()
            t = t[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2).contiguous()
            t = t.to(dtype=torch.float32).div(255.0)
            reference_results.append(frame_from_tensor(t, scale=1.0))

        for i, (pooled, ref, orig) in enumerate(zip(pooled_results, reference_results, frames)):
            assert (pooled == ref).all(), f"frame {i}: pooled path diverged from reference"
            assert (pooled == orig).all(), f"frame {i}: pooled path corrupted vs. source"
