"""Auto Video Fixer - Scene-based processing.

Orchestrates the scene-mode pipeline (see docs/REQUIREMENTS.md features 1-2 and
AGENTS.md's "Scene mode" section):

1. Split a video into per-scene clips at the boundaries already found by
   ``VideoAnalyzer.detect_events`` (re-encoded, not stream-copied -- see
   ``split_scene_video``'s docstring for why).
2. Optionally run per-scene VLM sampling + a coordinating text-LLM pass that
   flags scenes that aren't part of the video's main content, and drops them
   (``run_scene_coordinator`` in ``core/analysis.py``; fails open on any error).
3. Run stabilize (with per-scene strength tiering) and/or interpolate (never
   across a cut) on each kept scene's clip independently.
4. Concatenate the processed video clips and the original audio (cut at the
   same scene boundaries) back into one file for the remaining whole-video
   stages (upscale/denoise/deblock/normalize/encode) to continue from.

Scenes are processed in parallel (bounded worker pool, shared with the
traditional-interpolation chunk pool -- see ``_scene_worker_budget``) when more
than one scene needs processing; the AI/RIFE interpolation path stays serial
within that pool (GPU-bound, contends rather than parallelizes).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from autovideofixer.config import Config, resolve_timeout
from autovideofixer.core.analysis import SceneEvent, VideoAnalyzer, run_scene_coordinator
from autovideofixer.core.ffmpeg_utils import probe, run_ffmpeg
from autovideofixer.core.stages.base import StageStatus
from autovideofixer.core.stages.interpolate import InterpolateStage
from autovideofixer.core.stages.stabilize import StabilizeStage

logger = logging.getLogger(__name__)


@dataclass
class SceneModeResult:
    """Outcome of running the scene-mode preprocessing pipeline."""

    output_path: str
    total_scenes: int
    kept_scenes: int
    dropped_scenes: list[dict[str, Any]]
    stabilize_tiers: dict[int, str]  # scene_index -> "skip"|"normal"|"aggressive"
    interpolated_scenes: list[int]


def _run_ffmpeg_ok(args: list[str], timeout: float | None = None) -> bool:
    result = run_ffmpeg(args, timeout=timeout)
    return result.returncode == 0


def split_scene_video(
    input_path: str,
    scene: SceneEvent,
    out_path: str,
    crf: int = 14,
    preset: str = "veryfast",
    timeout: float | None = None,
) -> bool:
    """Extract one scene's video-only frames as an independently re-encoded clip.

    Re-encodes rather than stream-copies: a stream-copy cut (``-c copy``) snaps
    to the nearest keyframe, which for typical GOP sizes (every 2-10s) can drift
    a cut by up to a whole GOP -- unacceptable for a scene boundary that's
    meant to land exactly where scene detection found it, and exactly where the
    (separately re-cut, see split_scene_audio) audio segment starts too. Every
    scene segment gets re-encoded again anyway by the pipeline's own downstream
    "encode" stage after concat, so the quality cost of one extra re-encode
    pass here is not a new loss on top of an otherwise-lossless path.
    """
    args = [
        "-ss",
        str(scene.start_time),
        "-i",
        input_path,
        "-t",
        str(max(scene.duration, 0.01)),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-y",
        out_path,
    ]
    return _run_ffmpeg_ok(args, timeout=timeout)


def split_scene_audio(
    input_path: str, scene: SceneEvent, out_path: str, timeout: float | None = None
) -> bool:
    """Extract one scene's audio as an independently re-encoded segment.

    Re-encoded (not stream-copied) for the same reason as
    ``split_scene_video``: a stream-copy audio cut can only land on a codec
    frame boundary (e.g. every ~23ms for AAC), which is usually fine on its
    own, but re-encoding here keeps the audio cut points sample-accurate
    against the (also re-encoded, frame-accurate) video cut so concat doesn't
    introduce drift between them.
    """
    args = [
        "-ss",
        str(scene.start_time),
        "-i",
        input_path,
        "-t",
        str(max(scene.duration, 0.01)),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-y",
        out_path,
    ]
    return _run_ffmpeg_ok(args, timeout=timeout)


def _concat_demuxer(
    paths: list[str],
    out_path: str,
    list_path: str,
    extra_args: list[str],
    timeout: float | None = None,
) -> bool:
    with open(list_path, "w") as f:
        for p in paths:
            f.write(f"file '{p.replace(chr(39), chr(92) + chr(39))}'\n")
    args = ["-f", "concat", "-safe", "0", "-i", list_path, *extra_args, "-y", out_path]
    return _run_ffmpeg_ok(args, timeout=timeout)


def concat_video_clips(
    clip_paths: list[str], out_path: str, work_dir: str, timeout: float | None = None
) -> bool:
    """Concatenate video-only clips via the ffmpeg concat demuxer.

    Re-encodes during concat (rather than ``-c copy``) so minor
    codec-parameter mismatches between per-scene clips (e.g. a scene that
    skipped stabilization/interpolation vs. one that didn't) can't make the
    concat demuxer's stream-copy path fail outright -- acceptable since the
    pipeline's own "encode" stage re-encodes the reassembled file again anyway.
    """
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], out_path)
        return True
    list_path = os.path.join(work_dir, "concat_video_list.txt")
    return _concat_demuxer(
        clip_paths,
        out_path,
        list_path,
        ["-c:v", "libx264", "-crf", "14", "-preset", "veryfast", "-an"],
        timeout=timeout,
    )


def concat_audio_clips(
    clip_paths: list[str], out_path: str, work_dir: str, timeout: float | None = None
) -> bool:
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], out_path)
        return True
    list_path = os.path.join(work_dir, "concat_audio_list.txt")
    return _concat_demuxer(
        clip_paths, out_path, list_path, ["-c:a", "aac", "-b:a", "192k"], timeout=timeout
    )


def mux_video_audio(
    video_path: str, audio_path: str | None, out_path: str, timeout: float | None = None
) -> bool:
    if not audio_path:
        shutil.copy2(video_path, out_path)
        return True
    args = [
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        "-y",
        out_path,
    ]
    return _run_ffmpeg_ok(args, timeout=timeout)


def stabilize_scene_clip(
    clip_path: str,
    out_path: str,
    config: Config,
) -> tuple[bool, str]:
    """Run per-scene stabilization with strength tiering.

    Tiers (see ``scenes.stabilize.*`` config, ``Config.DEFAULTS``):
      - "skip": StabilizeStage's own shake analysis (avg_shake vs.
        ``stages.stabilize.threshold``) decided this scene doesn't need
        stabilization at all -- the clip passes through unchanged.
      - "normal": stabilized once with the stage's normal (config-default)
        smoothness.
      - "aggressive": avg_shake exceeded ``scenes.stabilize.
        aggressive_shake_threshold`` -- re-run with smoothness multiplied by
        ``scenes.stabilize.aggressive_smoothness_multiplier`` for a stronger
        correction than a uniform whole-video pass would have applied to this
        (unusually shaky) scene.

    Returns (ok, tier).
    """
    stage = StabilizeStage(config)
    probe_info = probe(clip_path)
    input_info = {"has_audio": probe_info.has_audio}

    result = stage.execute(clip_path, out_path, input_info=input_info)
    if result.status == StageStatus.FAILED:
        return False, "failed"

    avg_shake = float(result.metadata.get("avg_shake", 0.0))
    skipped = bool(result.metadata.get("skipped", False))
    if skipped:
        return True, "skip"

    tier_cfg = config.get("scenes", "stabilize", default={})
    if not tier_cfg.get("enabled", True):
        return True, "normal"

    aggressive_threshold = tier_cfg.get("aggressive_shake_threshold", 8.0)
    if avg_shake < aggressive_threshold:
        return True, "normal"

    multiplier = tier_cfg.get("aggressive_smoothness_multiplier", 2.0)
    base_smoothness = config.get("stages", "stabilize", "smoothness", default=40)
    aggressive_out = out_path + ".aggressive.mkv"
    result2 = stage.execute(
        clip_path,
        aggressive_out,
        input_info=input_info,
        smoothness=int(base_smoothness * multiplier),
    )
    if result2.status == StageStatus.FAILED:
        # Keep the normal-tier result rather than failing the scene outright.
        logger.warning(
            "Aggressive-tier re-stabilize failed for %s (avg_shake=%.2f); "
            "keeping normal-tier output",
            clip_path,
            avg_shake,
        )
        return True, "normal"

    try:
        os.replace(aggressive_out, out_path)
    except OSError:
        shutil.move(aggressive_out, out_path)
    return True, "aggressive"


def interpolate_scene_clip(
    clip_path: str,
    out_path: str,
    config: Config,
    target_fps: float | None,
    parallel_chunks_override: int | None = None,
) -> tuple[bool, bool]:
    """Run per-scene frame interpolation (never across a cut, by construction --
    this only ever sees a single scene's own clip).

    Returns (ok, ran) -- ``ran`` is False when the scene's own framerate is
    already at/above target (should_run() gate), in which case the clip
    passes through unchanged and ``ok`` is still True.
    """
    stage = InterpolateStage(config)
    probe_info = probe(clip_path)
    input_info = {
        "framerate": probe_info.framerate,
        "has_audio": probe_info.has_audio,
        # should_run() checks input_info["target_framerate"] first, falling
        # back to config quality.quality_target.target_framerate -- pass the
        # caller's target_fps directly so a scene-mode run isn't silently
        # skipped when the caller didn't also mirror it into config (e.g.
        # tests, or a caller that only has a local target_fps value).
        "target_framerate": target_fps,
    }
    should_run, _reason = stage.should_run(input_info)
    if not should_run:
        shutil.copy2(clip_path, out_path)
        return True, False

    use_ai = config.get("general", "use_ai", default=None)
    method = "ai" if use_ai else "traditional"
    kwargs: dict[str, Any] = {"target_fps": target_fps, "method": method}
    if method == "traditional" and parallel_chunks_override is not None:
        # Bound the per-scene chunk pool so concurrently-processed scenes don't
        # multiplicatively oversubscribe the machine's CPU budget (see
        # _scene_worker_budget / run_scene_pipeline).
        kwargs["parallel_chunks"] = parallel_chunks_override

    result = stage.execute(clip_path, out_path, input_info=input_info, **kwargs)
    return result.status != StageStatus.FAILED, True


def _scene_worker_budget(num_scenes: int, config: Config) -> tuple[int, int]:
    """Split the CPU budget between "scenes run concurrently" and "chunks per
    scene's traditional interpolation run concurrently" so the two pools don't
    oversubscribe each other.

    Returns (scene_workers, per_scene_parallel_chunks).
    """
    cpu = os.cpu_count() or 4
    total_budget = max(1, min(cpu, 8))
    if num_scenes <= 1:
        return 1, total_budget
    scene_workers = max(1, min(num_scenes, total_budget))
    per_scene_chunks = max(1, total_budget // scene_workers)
    return scene_workers, per_scene_chunks


def _collect_scene_vlm_summaries(
    input_path: str, scenes: list[SceneEvent], config: Config
) -> list[dict[str, Any]]:
    analyzer = VideoAnalyzer(config)
    summaries = []
    for idx, scene in enumerate(scenes):
        vlm_result = analyzer.run_vlm_analysis_for_scene(input_path, scene)
        summaries.append(
            {
                "index": idx,
                "start_time": scene.start_time,
                "end_time": scene.end_time,
                "duration": scene.duration,
                "summary": vlm_result.get("summary", ""),
                "tags": vlm_result.get("tags", []),
            }
        )
    return summaries


def run_scene_pipeline(
    input_path: str,
    config: Config,
    run_stabilize: bool,
    run_interpolate: bool,
    target_fps: float | None,
    work_dir: str | None = None,
) -> SceneModeResult | None:
    """Run the full scene-mode preprocessing pipeline for one job.

    Returns None if scene mode found nothing to do (fewer than 2 scenes, or
    scene detection itself failed) -- the caller should fall back to the
    normal whole-video pipeline in that case. Never raises: any internal
    failure is logged and treated the same as "nothing to do" so scene mode
    can never regress a job that would have succeeded without it.
    """
    own_work_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="avf_scenes_")

    try:
        analyzer = VideoAnalyzer(config)
        scenes = analyzer.detect_events(input_path)
        if len(scenes) < 2:
            logger.info(
                "Scene mode: only %d scene(s) detected for %s, nothing to split -- "
                "falling back to whole-video processing",
                len(scenes),
                input_path,
            )
            return None

        dropped_info: list[dict[str, Any]] = []
        kept_scenes = list(enumerate(scenes))

        if config.get("scenes", "drop_non_content", default=False):
            try:
                summaries = _collect_scene_vlm_summaries(input_path, scenes, config)
                coordinator_result = run_scene_coordinator(summaries, config)
            except Exception:
                logger.warning(
                    "Scene coordinator pass raised an exception; failing open -- "
                    "keeping all %d scene(s)",
                    len(scenes),
                    exc_info=True,
                )
                coordinator_result = {"drop": [], "reasons": {}, "failed": True}

            drop_indices = set(coordinator_result.get("drop", []))
            if drop_indices:
                for idx in sorted(drop_indices):
                    dropped_info.append(
                        {
                            "index": idx,
                            "start_time": scenes[idx].start_time,
                            "end_time": scenes[idx].end_time,
                            "reason": coordinator_result.get("reasons", {}).get(
                                str(idx), coordinator_result.get("reasons", {}).get(idx, "")
                            ),
                        }
                    )
                kept_scenes = [(i, s) for i, s in kept_scenes if i not in drop_indices]

        if not kept_scenes:
            logger.warning(
                "Scene coordinator flagged ALL scenes for drop -- refusing to produce an "
                "empty output; keeping every scene instead"
            )
            kept_scenes = list(enumerate(scenes))
            dropped_info = []

        has_audio = probe(input_path).has_audio
        scene_workers, per_scene_chunks = _scene_worker_budget(len(kept_scenes), config)
        intermediate_crf = config.get("scenes", "intermediate_crf", default=14)
        intermediate_preset = config.get("scenes", "intermediate_preset", default="veryfast")
        # Scene split/concat/mux passes scale with input length like any stage's
        # main pass, so they share pipeline.stage_timeout (null = unlimited)
        # rather than a fixed cap.
        ffmpeg_timeout = resolve_timeout(
            config.get("pipeline", "stage_timeout", default=None), "pipeline.stage_timeout"
        )

        tiers: dict[int, str] = {}
        interpolated: list[int] = []
        video_clip_paths: dict[int, str] = {}
        audio_clip_paths: dict[int, str] = {}
        failures: list[str] = []

        def _process_one(pair: tuple[int, SceneEvent]) -> tuple[int, bool]:
            idx, scene = pair
            raw_video = os.path.join(work_dir, f"scene_{idx:04d}_raw.mp4")
            if not split_scene_video(
                input_path,
                scene,
                raw_video,
                crf=intermediate_crf,
                preset=intermediate_preset,
                timeout=ffmpeg_timeout,
            ):
                return idx, False

            current = raw_video
            if run_stabilize:
                stab_out = os.path.join(work_dir, f"scene_{idx:04d}_stab.mkv")
                ok, tier = stabilize_scene_clip(current, stab_out, config)
                if not ok:
                    return idx, False
                tiers[idx] = tier
                current = stab_out

            if run_interpolate:
                interp_out = os.path.join(work_dir, f"scene_{idx:04d}_interp.mp4")
                ok, ran = interpolate_scene_clip(
                    current,
                    interp_out,
                    config,
                    target_fps,
                    parallel_chunks_override=per_scene_chunks,
                )
                if not ok:
                    return idx, False
                if ran:
                    interpolated.append(idx)
                current = interp_out

            video_clip_paths[idx] = current

            if has_audio:
                audio_out = os.path.join(work_dir, f"scene_{idx:04d}_audio.m4a")
                if not split_scene_audio(input_path, scene, audio_out, timeout=ffmpeg_timeout):
                    return idx, False
                audio_clip_paths[idx] = audio_out

            return idx, True

        if scene_workers <= 1:
            for pair in kept_scenes:
                idx, ok = _process_one(pair)
                if not ok:
                    failures.append(str(idx))
        else:
            with ThreadPoolExecutor(max_workers=scene_workers) as executor:
                futures = {executor.submit(_process_one, pair): pair[0] for pair in kept_scenes}
                for future in as_completed(futures):
                    idx, ok = future.result()
                    if not ok:
                        failures.append(str(idx))

        if failures:
            logger.warning(
                "Scene mode: %d scene(s) failed to process (%s); aborting scene mode for "
                "this job, falling back to whole-video processing",
                len(failures),
                ", ".join(failures),
            )
            return None

        ordered_indices = [idx for idx, _ in kept_scenes]
        video_clips = [video_clip_paths[i] for i in ordered_indices]
        concat_video = os.path.join(work_dir, "concat_video.mp4")
        if not concat_video_clips(video_clips, concat_video, work_dir, timeout=ffmpeg_timeout):
            logger.warning("Scene mode: video concat failed; falling back to whole-video")
            return None

        final_out = os.path.join(work_dir, "scene_mode_output.mp4")
        if has_audio:
            audio_clips = [audio_clip_paths[i] for i in ordered_indices]
            concat_audio = os.path.join(work_dir, "concat_audio.m4a")
            if not concat_audio_clips(audio_clips, concat_audio, work_dir, timeout=ffmpeg_timeout):
                logger.warning("Scene mode: audio concat failed; falling back to whole-video")
                return None
            if not mux_video_audio(concat_video, concat_audio, final_out, timeout=ffmpeg_timeout):
                logger.warning("Scene mode: final mux failed; falling back to whole-video")
                return None
        else:
            shutil.copy2(concat_video, final_out)

        if own_work_dir:
            # Move the final output outside work_dir before the `finally`
            # below removes work_dir -- otherwise the returned path would be
            # deleted out from under the caller before they can use it.
            import uuid

            outside_dir = tempfile.gettempdir()
            moved_out = os.path.join(outside_dir, f".avf_scene_output_{uuid.uuid4().hex[:8]}.mp4")
            shutil.move(final_out, moved_out)
            final_out = moved_out

        logger.info(
            "Scene mode: processed %d/%d scene(s) (%d dropped), stabilize tiers=%s, "
            "interpolated=%s",
            len(kept_scenes),
            len(scenes),
            len(dropped_info),
            {k: v for k, v in tiers.items()},
            interpolated,
        )

        return SceneModeResult(
            output_path=final_out,
            total_scenes=len(scenes),
            kept_scenes=len(kept_scenes),
            dropped_scenes=dropped_info,
            stabilize_tiers=tiers,
            interpolated_scenes=interpolated,
        )
    except Exception:
        logger.warning(
            "Scene mode preprocessing raised an unexpected exception for %s; falling back "
            "to whole-video processing",
            input_path,
            exc_info=True,
        )
        return None
    finally:
        # Only remove work_dir when we created it ourselves -- a caller-provided
        # work_dir is the caller's to manage. The final output was already moved
        # outside work_dir (see above) before this runs, on the success path.
        if own_work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
