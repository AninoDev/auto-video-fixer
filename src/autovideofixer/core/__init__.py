"""Auto Video Fixer - Core package."""

from autovideofixer.core.stages import (
    BaseStage,
    DeblockStage,
    DenoiseVideoStage,
    DetectStage,
    EncodeStage,
    HDRStage,
    InterpolateStage,
    NormalizeAudioStage,
    NormalizeVolumeStage,
    RemuxStage,
    SpeedStage,
    StabilizeStage,
    StageResult,
    StageStatus,
    UpscaleStage,
)

__all__ = [
    "BaseStage",
    "StageResult",
    "StageStatus",
    "DetectStage",
    "StabilizeStage",
    "DeblockStage",
    "DenoiseVideoStage",
    "UpscaleStage",
    "InterpolateStage",
    "NormalizeVolumeStage",
    "NormalizeAudioStage",
    "EncodeStage",
    "RemuxStage",
    "SpeedStage",
    "HDRStage",
]
