"""Pulse feature exports.

The implementation currently obtains pulse-transition records from finite
branch maps.  This file exists as a stable import location for later primitive
pulse builders based on detuning, Rabi amplitude, polarization, and duration.
"""
from .spectroscopy import (
    PULSE_FEATURE_DIM,
    PULSE_TRANSITION_DIM,
    build_pulse_library_from_branch_model,
)

__all__ = [
    "PULSE_FEATURE_DIM",
    "PULSE_TRANSITION_DIM",
    "build_pulse_library_from_branch_model",
]
