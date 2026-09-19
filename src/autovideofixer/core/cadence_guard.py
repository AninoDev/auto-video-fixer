"""Auto Video Fixer - raw-pipe frame-count vs. cadence guard (REQUIREMENTS.md § 19.1).

**The incident this guards against.** A user's 23.8s clip came out 72 seconds long and
badly stuttery, and nothing in the pipeline noticed -- every stage reported COMPLETED,
the run exited 0, all gates passed, and the JSON report looked healthy. Diagnosis: the
installed ``avf_framepipe`` binary was a stale cached wheel built before the § 12.4b
``-fps_mode passthrough`` reader fix, so the raw-frame reader re-expanded a retimed VFR
intermediate from 235 frames back to 714 CFR frames. A raw-pipe writer then tagged those
714 frames at ``true_framerate`` = 9.864fps -- the *recovered* cadence, which was only
correct for the 235-frame stream it was computed from -- giving 714 / 9.864 = 72.4s.

The stale wheel was the trigger and rebuilding fixed it (see § 12.6's build gotcha), but
the deeper failure is that a **detectable** inconsistency (frame count and cadence
disagreed by 3x) produced a silently wrong deliverable. This module is the tripwire:
before a raw-pipe stage (``upscale``, ``deblock``, ``denoise_video`` via the shared
Real-ESRGAN wrapper, ``interpolate``'s AI path, ``stabilize``'s raw pipe) tags its output
with a carrier framerate, it must check ``actual_frames`` against what the candidate
carrier fps implies for the known input duration, and refuse to trust a carrier that
disagrees.

Pure and unit-testable on purpose: no ffmpeg process, no GPU, no video file -- just
frame counts, a duration, and a candidate fps.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default disagreement tolerance for resolve_cadence(): a *fraction* of the
# expected frame count (e.g. 0.15 = 15%). Deliberately looser than
# `pipeline.duration_guard_tolerance` (5%, see core/pipeline.py's duration
# invariant check / REQUIREMENTS.md § 19.2): frame-count estimates from
# container metadata can legitimately be off by a frame or two at the start/
# end of a clip (partial frame rounding, decoder priming), so this guard
# should only fire on the class of GROSS (multi-x) disagreement the incident
# above exhibits, not on ordinary rounding noise -- that's what the cheaper,
# stricter § 19.2 duration check is for.
DEFAULT_CADENCE_GUARD_TOLERANCE = 0.15


@dataclass
class CadenceResolution:
    """Result of ``resolve_cadence()``.

    ``resolved_fps`` is always the fps a caller should actually use to tag
    its output: the untouched ``candidate_fps`` when consistent (or when the
    check couldn't run at all -- see ``resolve_cadence``'s zero/unknown-
    duration handling), or ``actual_frames / input_duration`` -- a rate
    derived from what actually came through, self-consistent by
    construction -- when a mismatch was detected.
    """

    resolved_fps: float
    mismatch: bool
    reason: str | None = None
    expected_frames: float | None = None
    actual_frames: int = 0


def resolve_cadence(
    actual_frames: int,
    input_duration: float | None,
    candidate_fps: float | None,
    tolerance: float = DEFAULT_CADENCE_GUARD_TOLERANCE,
) -> CadenceResolution:
    """Check ``actual_frames`` against what ``candidate_fps`` implies for
    ``input_duration``, per REQUIREMENTS.md § 19.1:
    ``expected_frames ~= input_duration * carrier_fps``.

    Returns a ``CadenceResolution`` whose ``resolved_fps`` a raw-pipe stage
    should tag its output with instead of blindly trusting
    ``candidate_fps`` -- see the module docstring for why.

    Deliberately fails open (``mismatch=False``, ``resolved_fps=candidate_fps``)
    whenever the check can't be meaningfully run, rather than raising or
    guessing:

    - ``candidate_fps`` missing/<=0 -- nothing to compare against.
    - ``input_duration`` missing/<=0 -- VFR containers in particular often
      report an unreliable or absent duration (REQUIREMENTS.md § 12.3); a
      guard that fires on that is a false alarm, not a catch.
    - ``actual_frames`` <=0 -- a genuinely empty output is a different
      failure mode (stages already fail outright on "no frames produced"),
      not a cadence disagreement.
    """
    if candidate_fps is None or candidate_fps <= 0:
        return CadenceResolution(
            resolved_fps=candidate_fps or 0.0, mismatch=False, actual_frames=actual_frames
        )
    if input_duration is None or input_duration <= 0:
        return CadenceResolution(
            resolved_fps=candidate_fps, mismatch=False, actual_frames=actual_frames
        )
    if actual_frames <= 0:
        return CadenceResolution(
            resolved_fps=candidate_fps, mismatch=False, actual_frames=actual_frames
        )

    expected_frames = input_duration * candidate_fps
    if expected_frames <= 0:
        return CadenceResolution(
            resolved_fps=candidate_fps,
            mismatch=False,
            expected_frames=expected_frames,
            actual_frames=actual_frames,
        )

    ratio = actual_frames / expected_frames
    if abs(ratio - 1.0) <= tolerance:
        return CadenceResolution(
            resolved_fps=candidate_fps,
            mismatch=False,
            expected_frames=expected_frames,
            actual_frames=actual_frames,
        )

    resolved_fps = actual_frames / input_duration
    reason = (
        f"actual frame count {actual_frames} disagrees with the candidate carrier rate "
        f"{candidate_fps:.4f}fps over an input duration of {input_duration:.3f}s "
        f"(expected ~{expected_frames:.1f} frames, got {actual_frames}, "
        f"{ratio:.2f}x expected); likely a VFR stream re-expanded back to CFR by the "
        "raw-pipe reader -- on a mixed Python/Rust install, check for a stale "
        "avf_framepipe build missing the -fps_mode passthrough reader fix "
        "(REQUIREMENTS.md § 12.6: `uv sync --all-extras --reinstall-package avf_framepipe`). "
        f"Deriving the carrier from actual frames instead: {resolved_fps:.4f}fps."
    )
    return CadenceResolution(
        resolved_fps=resolved_fps,
        mismatch=True,
        reason=reason,
        expected_frames=expected_frames,
        actual_frames=actual_frames,
    )


def cadence_metadata(resolution: CadenceResolution) -> dict[str, object]:
    """Stage-metadata fields for ``resolution``, per REQUIREMENTS.md § 19.1's
    "record cadence_mismatch: true (plus both counts) in stage metadata".

    Returns ``{}`` when there was no mismatch -- callers merge this
    unconditionally into their StageResult.metadata dict without an `if`.
    """
    if not resolution.mismatch:
        return {}
    return {
        "cadence_mismatch": True,
        "cadence_expected_frames": resolution.expected_frames,
        "cadence_actual_frames": resolution.actual_frames,
        "cadence_resolved_fps": resolution.resolved_fps,
    }


__all__ = [
    "DEFAULT_CADENCE_GUARD_TOLERANCE",
    "CadenceResolution",
    "resolve_cadence",
    "cadence_metadata",
]
