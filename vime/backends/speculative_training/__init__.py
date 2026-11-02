"""Online training support for external speculative draft models."""

from .config import external_draft_enabled, should_run_draft_interval, validate_external_draft_args

__all__ = [
    "DraftFeatureSample",
    "VersionedFeatureQueue",
    "external_draft_enabled",
    "should_run_draft_interval",
    "validate_external_draft_args",
]


def __getattr__(name):
    if name in {"DraftFeatureSample", "VersionedFeatureQueue"}:
        from .feature_schema import DraftFeatureSample, VersionedFeatureQueue

        values = {
            "DraftFeatureSample": DraftFeatureSample,
            "VersionedFeatureQueue": VersionedFeatureQueue,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
