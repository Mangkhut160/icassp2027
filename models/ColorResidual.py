"""Model-package exports for the optional color residual decoder."""

from color_residual_core import (
    ColorResidualDecoder,
    ColorResidualHead,
    GoalConditionedColorResidualDecoder,
    GoalConditionedColorResidualHead,
    attach_color_residual_decoder,
)

__all__ = [
    'ColorResidualDecoder',
    'ColorResidualHead',
    'GoalConditionedColorResidualDecoder',
    'GoalConditionedColorResidualHead',
    'attach_color_residual_decoder',
]
