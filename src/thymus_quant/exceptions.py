from __future__ import annotations

"""Package-specific exceptions and warnings."""


class ThymusQuantError(Exception):
    """Base class for thymus-quant errors."""


class InputValidationError(ThymusQuantError, ValueError):
    """Raised when user input does not satisfy the public data contract."""


class MissingGeometryError(InputValidationError):
    """Raised when spacing/geometry required for volume computation is missing."""


class SegmentorConfigurationError(InputValidationError):
    """Raised when a segmentor alias, member, or weight configuration is invalid."""


class QuantificationError(ThymusQuantError):
    """Raised when quantification cannot be computed."""


class NumericalError(QuantificationError):
    """Raised when a numerical subroutine fails."""


class ExperimentalWarning(UserWarning):
    """Warning emitted when an experimental method is used."""
