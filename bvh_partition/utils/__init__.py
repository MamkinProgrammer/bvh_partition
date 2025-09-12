"""
Утилиты для BVH разбиения
"""
from .density import (
    estimate_xy_density,
    estimate_knn_spacing,
    estimate_scale,
    compute_density_stats
)

__all__ = [
    'estimate_xy_density',
    'estimate_knn_spacing', 
    'estimate_scale',
    'compute_density_stats'
]