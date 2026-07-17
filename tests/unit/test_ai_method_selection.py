"""Tests for Fix 3: per-stage AI selection (BaseStage.resolve_ai_method),
loud method-selection logging, scene-mode parity (scenes.interpolate.use_ai),
and the GPU inference semaphore (gpu.max_concurrent_inferences).
"""

from __future__ import annotations

import threading
import time

from autovideofixer.ai.torch_utils import get_gpu_inference_semaphore
from autovideofixer.config import Config
from autovideofixer.core.stages.base import BaseStage


class _DummyAIStage(BaseStage):
    """Minimal concrete stage for exercising resolve_ai_method() /
    _log_ai_method_choice() in isolation."""

    name = "dummy_ai_stage"
    display_name = "Dummy AI Stage"
    category = "enhancement"

    def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
        raise NotImplementedError


class TestResolveAiMethod:
    def test_auto_default_wins_with_nothing_set(self, tmp_path):
        stage = _DummyAIStage(Config(tmp_path / "c.yaml"))
        method, source = stage.resolve_ai_method(None, "traditional")
        assert method == "traditional"
        assert source == "auto default"

        method, source = stage.resolve_ai_method(None, "ai")
        assert method == "ai"
        assert source == "auto default"

    def test_general_use_ai_overrides_auto_default(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(True, "general", "use_ai")
        stage = _DummyAIStage(config)
        method, source = stage.resolve_ai_method(None, "traditional")
        assert method == "ai"
        assert source == "--ai"

        config2 = Config(tmp_path / "c2.yaml")
        config2.set(False, "general", "use_ai")
        stage2 = _DummyAIStage(config2)
        method, source = stage2.resolve_ai_method(None, "ai")
        assert method == "traditional"
        assert source == "--no-ai"

    def test_per_stage_use_ai_overrides_general(self, tmp_path):
        """Per the ai_fallback convention this mirrors: the global CLI flag
        does NOT override an explicit per-stage config value."""
        config = Config(tmp_path / "c.yaml")
        config.set(True, "general", "use_ai")
        config.set(False, "stages", "dummy_ai_stage", "use_ai")
        stage = _DummyAIStage(config)
        method, source = stage.resolve_ai_method(None, "ai")
        assert method == "traditional"
        assert source == "stages.dummy_ai_stage.use_ai: false"

        config2 = Config(tmp_path / "c2.yaml")
        config2.set(False, "general", "use_ai")
        config2.set(True, "stages", "dummy_ai_stage", "use_ai")
        stage2 = _DummyAIStage(config2)
        method, source = stage2.resolve_ai_method(None, "traditional")
        assert method == "ai"
        assert source == "stages.dummy_ai_stage.use_ai: true"

    def test_explicit_method_wins_outright(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(True, "general", "use_ai")
        config.set(False, "stages", "dummy_ai_stage", "use_ai")
        stage = _DummyAIStage(config)
        method, source = stage.resolve_ai_method("traditional", "ai")
        assert method == "traditional"
        assert source == "explicit method= argument"


class TestLogAiMethodChoice:
    def test_traditional_auto_default_includes_opt_in_hint(self, tmp_path, caplog):
        stage = _DummyAIStage(Config(tmp_path / "c.yaml"))
        with caplog.at_level("INFO"):
            stage._log_ai_method_choice(
                "traditional",
                "auto default",
                ai_desc="AI interpolation (RIFE 'rife_v4.6', backend torch)",
                traditional_desc="traditional minterpolate",
                ai_hint="AI/RIFE model 'rife_v4.6'",
            )
        messages = [r.message for r in caplog.records]
        assert any("using traditional minterpolate" in m for m in messages)
        assert any("set stages.dummy_ai_stage.use_ai: true or pass --ai" in m for m in messages)
        assert any("AI/RIFE model 'rife_v4.6'" in m for m in messages)

    def test_traditional_explicit_config_has_no_opt_in_hint(self, tmp_path, caplog):
        stage = _DummyAIStage(Config(tmp_path / "c.yaml"))
        with caplog.at_level("INFO"):
            stage._log_ai_method_choice(
                "traditional",
                "stages.dummy_ai_stage.use_ai: false",
                ai_desc="AI interpolation (...)",
                traditional_desc="traditional minterpolate",
                ai_hint="AI/RIFE model 'rife_v4.6'",
            )
        messages = [r.message for r in caplog.records]
        assert any(
            "using traditional minterpolate (stages.dummy_ai_stage.use_ai: false)" in m
            for m in messages
        )
        assert not any("pass --ai" in m for m in messages)

    def test_ai_chosen_includes_source(self, tmp_path, caplog):
        stage = _DummyAIStage(Config(tmp_path / "c.yaml"))
        with caplog.at_level("INFO"):
            stage._log_ai_method_choice(
                "ai",
                "stages.dummy_ai_stage.use_ai: true",
                ai_desc="AI interpolation (RIFE 'rife_v4.6', backend torch)",
                traditional_desc="traditional minterpolate",
                ai_hint="AI/RIFE model 'rife_v4.6'",
            )
        messages = [r.message for r in caplog.records]
        assert any(
            "using AI interpolation (RIFE 'rife_v4.6', backend torch) "
            "(selected by stages.dummy_ai_stage.use_ai: true)" in m
            for m in messages
        )


class TestFourStagesAutoDefaults:
    """The four AI-capable stages' hardcoded auto defaults are deliberate,
    not oversights (per the spec's ADJUSTED Fix 3): upscale/deblock -> ai,
    denoise_video/interpolate -> traditional."""

    def test_upscale_auto_default_traditional_when_already_at_target(self, tmp_path):
        from autovideofixer.core.stages.upscale import UpscaleStage

        stage = UpscaleStage(Config(tmp_path / "c.yaml"))
        stage._input_info = {"resolution": (1920, 1080)}
        assert stage._auto_method_for_scale(1920, 1080) == "traditional"

    def test_upscale_auto_default_ai_when_upscale_needed(self, tmp_path):
        from autovideofixer.core.stages.upscale import UpscaleStage

        stage = UpscaleStage(Config(tmp_path / "c.yaml"))
        stage._input_info = {"resolution": (960, 540)}
        assert stage._auto_method_for_scale(1920, 1080) == "ai"

    def test_deblock_resolve_ai_method_defaults_to_ai(self, tmp_path):
        from autovideofixer.core.stages.deblock import DeblockStage

        stage = DeblockStage(Config(tmp_path / "c.yaml"))
        method, source = stage.resolve_ai_method(None, "ai")
        assert method == "ai"
        assert source == "auto default"

    def test_denoise_video_resolve_ai_method_defaults_to_traditional(self, tmp_path):
        from autovideofixer.core.stages.denoise_video import DenoiseVideoStage

        stage = DenoiseVideoStage(Config(tmp_path / "c.yaml"))
        method, source = stage.resolve_ai_method(None, "traditional")
        assert method == "traditional"
        assert source == "auto default"

    def test_interpolate_resolve_ai_method_defaults_to_traditional(self, tmp_path):
        from autovideofixer.core.stages.interpolate import InterpolateStage

        stage = InterpolateStage(Config(tmp_path / "c.yaml"))
        method, source = stage.resolve_ai_method(None, "traditional")
        assert method == "traditional"
        assert source == "auto default"

    def test_per_stage_use_ai_true_flips_denoise_video_default(self, tmp_path):
        from autovideofixer.core.stages.denoise_video import DenoiseVideoStage

        config = Config(tmp_path / "c.yaml")
        config.set(True, "stages", "denoise_video", "use_ai")
        stage = DenoiseVideoStage(config)
        method, source = stage.resolve_ai_method(None, "traditional")
        assert method == "ai"
        assert source == "stages.denoise_video.use_ai: true"

    def test_per_stage_use_ai_false_flips_deblock_default(self, tmp_path):
        from autovideofixer.core.stages.deblock import DeblockStage

        config = Config(tmp_path / "c.yaml")
        config.set(False, "stages", "deblock", "use_ai")
        stage = DeblockStage(config)
        method, source = stage.resolve_ai_method(None, "ai")
        assert method == "traditional"
        assert source == "stages.deblock.use_ai: false"


class TestSceneModeInterpolateUseAiParity:
    def test_scene_override_null_mirrors_standard_stage_resolution(self, tmp_path):
        """scenes.interpolate.use_ai: null (default) must resolve with the
        exact same precedence as the standard interpolate stage."""
        from autovideofixer.core.stages.interpolate import InterpolateStage

        config = Config(tmp_path / "c.yaml")
        config.set(True, "stages", "interpolate", "use_ai")
        stage = InterpolateStage(config)
        # This is exactly what interpolate_scene_clip does when
        # scenes.interpolate.use_ai is null.
        method, _source = stage.resolve_ai_method(None, "traditional")
        assert method == "ai"

    def test_scene_override_true_forces_ai_regardless_of_stage_config(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(False, "stages", "interpolate", "use_ai")
        config.set(True, "scenes", "interpolate", "use_ai")
        scene_override = config.get("scenes", "interpolate", "use_ai", default=None)
        assert scene_override is True

    def test_scene_override_false_forces_traditional_regardless_of_stage_config(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(True, "stages", "interpolate", "use_ai")
        config.set(False, "scenes", "interpolate", "use_ai")
        scene_override = config.get("scenes", "interpolate", "use_ai", default=None)
        assert scene_override is False


class TestGpuInferenceSemaphore:
    def test_default_limit_is_one(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        sem = get_gpu_inference_semaphore(config)
        # A limit-1 semaphore: one acquire succeeds, a second (non-blocking)
        # does not.
        assert sem.acquire(blocking=False) is True
        assert sem.acquire(blocking=False) is False
        sem.release()

    def test_limit_bounds_concurrency_across_threads(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(1, "gpu", "max_concurrent_inferences")
        sem = get_gpu_inference_semaphore(config)

        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        def _fake_inference():
            nonlocal max_concurrent, current
            sem.acquire()
            try:
                with lock:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                time.sleep(0.05)
                with lock:
                    current -= 1
            finally:
                sem.release()

        threads = [threading.Thread(target=_fake_inference) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert max_concurrent == 1

    def test_higher_limit_allows_more_concurrency(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(3, "gpu", "max_concurrent_inferences")
        sem = get_gpu_inference_semaphore(config)

        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        def _fake_inference():
            nonlocal max_concurrent, current
            sem.acquire()
            try:
                with lock:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                time.sleep(0.05)
                with lock:
                    current -= 1
            finally:
                sem.release()

        threads = [threading.Thread(target=_fake_inference) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert max_concurrent == 3


class TestSceneModeGpuSemaphoreGating:
    """Fix 3d, end-to-end through interpolate_scene_clip: two scene threads
    both resolving to the AI path must never run concurrent inference when
    gpu.max_concurrent_inferences is 1 -- the traditional/chunked path is
    NOT gated (see the else branch, ungated)."""

    def test_two_scene_threads_serialize_through_ai_path(self, tmp_path, monkeypatch):
        from autovideofixer.core import scenes as scenes_mod

        config = Config(tmp_path / "c.yaml")
        config.set(True, "scenes", "interpolate", "use_ai")
        config.set(1, "gpu", "max_concurrent_inferences")

        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        class _FakeStage:
            def __init__(self, _config):
                pass

            def should_run(self, _input_info):
                return True, None

            def execute(self, _input_path, _output_path, input_info=None, **kwargs):
                nonlocal max_concurrent, current
                with lock:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                time.sleep(0.05)
                with lock:
                    current -= 1
                from autovideofixer.core.stages.base import StageResult, StageStatus

                return StageResult(status=StageStatus.COMPLETED, metadata={})

        fake_probe_result = type(
            "P", (), {"framerate": 30.0, "has_audio": False, "resolution": (1920, 1080)}
        )()

        results = []

        def _run_one(idx):
            ok, ran = scenes_mod.interpolate_scene_clip(
                f"clip_{idx}.mp4",
                str(tmp_path / f"out_{idx}.mp4"),
                config,
                target_fps=60.0,
            )
            results.append((ok, ran))

        with (
            monkeypatch.context() as m,
        ):
            m.setattr(scenes_mod, "InterpolateStage", _FakeStage)
            m.setattr(scenes_mod, "probe", lambda _path: fake_probe_result)

            threads = [threading.Thread(target=_run_one, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        assert max_concurrent == 1
        assert all(ok and ran for ok, ran in results)
