"""Explicit exception types; failures that must never be silently swallowed."""


class PGOFPError(Exception):
    """Base class."""


class ConfigError(PGOFPError):
    """Incompatible / invalid configuration combination."""


class PriorUnavailableError(PGOFPError):
    """Prior artifact missing, mismatched or invalid."""


class NumericalError(PGOFPError):
    """NaN/Inf or failed numerical invariant inside an update."""


class ResumeIncompatibleError(PGOFPError):
    """Checkpoint cannot be resumed with the current configuration."""


class BackendUnavailableError(PGOFPError):
    """External backend (e.g. HFM SDK) is not installed or not verified."""


class UnsupportedCombinationError(PGOFPError):
    """Task/prior/algorithm combination that is intentionally not supported."""
