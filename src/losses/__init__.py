"""Loss functions module."""

from .spectral_loss import (
    SpectralNetLoss,
    OrthogonalityLoss,
    MMDLoss,
    SinkhornDivergence,
    CombinedSpectralLoss
)

__all__ = [
    'SpectralNetLoss',
    'OrthogonalityLoss',
    'MMDLoss',
    'SinkhornDivergence',
    'CombinedSpectralLoss'
]