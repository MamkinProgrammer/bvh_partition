"""
Утилиты для оценки плотности и пространственных характеристик
"""
import numpy as np
from typing import Tuple, Optional
import logging

from ..config import (
    DENSITY_CELL_SIZE,
    KNN_K,
    KNN_SAMPLE_SIZE,
    DEFAULT_GRID_CELLS,
    SCALE_EPSILON,
    VOLUME_EPSILON
)

logger = logging.getLogger(__name__)


def estimate_xy_density(points: np.ndarray, 
                        cell_size: float = DENSITY_CELL_SIZE) -> Tuple[float, float]:
    """
    Оценка 2D плотности (точек/м²) и среднего шага по XY
    
    Args:
        points: Облако точек (N x 3)
        cell_size: Размер ячейки для подсчёта занятости (м)
    
    Returns:
        (rho_2d, spacing_xy) - плотность и средний шаг
    """
    if points.shape[0] == 0:
        return 0.0, 1.0
    
    xy = points[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    dims = np.maximum(max_xy - min_xy, VOLUME_EPSILON)
    
    # Создаём сетку
    nx = max(1, int(np.ceil(dims[0] / cell_size)))
    ny = max(1, int(np.ceil(dims[1] / cell_size)))
    
    # Индексы ячеек для каждой точки
    ix = np.minimum(((xy[:, 0] - min_xy[0]) / cell_size).astype(np.int64), nx - 1)
    iy = np.minimum(((xy[:, 1] - min_xy[1]) / cell_size).astype(np.int64), ny - 1)
    
    # Подсчёт занятых ячеек
    occupied_cells = np.unique(ix * ny + iy).size
    area = occupied_cells * (cell_size * cell_size)
    area = max(area, VOLUME_EPSILON)
    
    rho_2d = points.shape[0] / area
    spacing_xy = 1.0 / np.sqrt(rho_2d)
    
    logger.debug(f"XY density: {rho_2d:.2f} pts/m², spacing: {spacing_xy:.3f} m")
    
    return rho_2d, spacing_xy


def estimate_knn_spacing(points: np.ndarray,
                        k: int = KNN_K,
                        sample_size: int = KNN_SAMPLE_SIZE) -> float:
    """
    Оценка среднего расстояния по k-ближайшим соседям
    
    Args:
        points: Облако точек (N x 3)
        k: Количество соседей
        sample_size: Размер выборки для ускорения
    
    Returns:
        Медианное расстояние до k-го соседа
    """
    n = points.shape[0]
    
    if n <= k + 1:
        # Недостаточно точек - грубая оценка
        bbox_diag = np.linalg.norm(points.ptp(points[:, :3], axis=0))
        return float(bbox_diag / 1000.0 + SCALE_EPSILON)
    
    # Подвыборка для ускорения
    sample_size = min(sample_size, n)
    rng = np.random.default_rng(42)  # Фиксированный seed для воспроизводимости
    idx = rng.choice(n, size=sample_size, replace=False)
    subsample = points[idx]
    
    try:
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=min(k + 1, subsample.shape[0]), algorithm='auto')
        nn.fit(subsample)
        distances, _ = nn.kneighbors(subsample, return_distance=True)
        
        # distances[:, 0] = 0 (расстояние до себя), берём k-тое
        if distances.shape[1] > k:
            spacing = np.median(distances[:, k])
        else:
            spacing = np.median(distances[:, -1])
        
        logger.debug(f"kNN spacing (k={k}): {spacing:.3f} m")
        return float(spacing)
        
    except ImportError:
        logger.warning("sklearn not available, using volume-based estimate")
        
        # Фолбэк: оценка по объёму
        volume = np.prod(points.max(axis=0) - points.min(axis=0) + VOLUME_EPSILON)
        rho_3d = n / max(volume, VOLUME_EPSILON)
        spacing = (1.0 / max(rho_3d, VOLUME_EPSILON)) ** (1.0/3.0)
        
        return float(spacing)


def estimate_scale(points: np.ndarray,
                  mode: str,
                  fixed_scale: Optional[float] = None,
                  kappa: float = 15.0) -> float:
    """
    Оценка масштаба сетки по указанному методу
    
    Args:
        points: Облако точек
        mode: Режим оценки ('auto', 'xy', 'knn', 'fixed')
        fixed_scale: Фиксированный масштаб для режима 'fixed'
        kappa: Коэффициент масштабирования
    
    Returns:
        Масштаб сетки в метрах на ячейку
    """
    if mode == 'fixed':
        if fixed_scale is None or fixed_scale <= 0:
            raise ValueError("Fixed scale must be positive")
        scale = fixed_scale
        
    elif mode == 'xy':
        # По 2D плотности
        _, spacing = estimate_xy_density(points)
        scale = max(spacing / kappa, SCALE_EPSILON)
        
    elif mode == 'knn':
        # По k-ближайшим соседям
        spacing = estimate_knn_spacing(points)
        scale = max(spacing / kappa, SCALE_EPSILON)
        
    elif mode == 'auto':
        # Автоматически по размеру bbox
        bbox = points.max(axis=0) - points.min(axis=0)
        max_dim = float(np.max(bbox))
        scale = max(max_dim / DEFAULT_GRID_CELLS, SCALE_EPSILON)
        
    else:
        raise ValueError(f"Unknown scale mode: {mode}")
    
    logger.info(f"Grid scale ({mode}): {scale:.6f} m/cell")
    return scale


def compute_density_stats(points: np.ndarray,
                          scale_m: float) -> dict:
    """
    Вычисление статистики плотности для облака точек
    
    Args:
        points: Облако точек
        scale_m: Масштаб сетки
    
    Returns:
        Словарь со статистикой
    """
    n = points.shape[0]
    
    # Bounding box
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    # Объёмы
    volume_m3 = np.prod(bbox_size)
    area_xy_m2 = bbox_size[0] * bbox_size[1]
    area_xz_m2 = bbox_size[0] * bbox_size[2]
    area_yz_m2 = bbox_size[1] * bbox_size[2]
    
    # Плотности
    rho_3d = n / max(volume_m3, VOLUME_EPSILON)
    rho_xy = n / max(area_xy_m2, VOLUME_EPSILON)
    rho_xz = n / max(area_xz_m2, VOLUME_EPSILON)
    rho_yz = n / max(area_yz_m2, VOLUME_EPSILON)
    
    # Оценка реальной 2D плотности через занятость
    real_rho_xy, real_spacing = estimate_xy_density(points)
    
    # Количество ячеек сетки
    grid_cells = np.ceil(bbox_size / scale_m).astype(int)
    total_cells = np.prod(grid_cells)
    
    return {
        'n_points': n,
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'bbox_size': bbox_size.tolist(),
        'volume_m3': volume_m3,
        'density_3d': rho_3d,
        'density_xy': rho_xy,
        'density_xz': rho_xz,
        'density_yz': rho_yz,
        'density_xy_occupied': real_rho_xy,
        'spacing_xy': real_spacing,
        'scale_m': scale_m,
        'grid_cells': grid_cells.tolist(),
        'total_grid_cells': int(total_cells)
    }
