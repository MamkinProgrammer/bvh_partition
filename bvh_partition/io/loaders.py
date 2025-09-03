"""
Загрузчики для различных форматов облаков точек
"""
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_point_cloud(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Универсальный загрузчик облака точек
    
    Args:
        file_path: Путь к файлу
    
    Returns:
        (coords, metadata)
        coords: Массив координат N×3 (x,y,z) в метрах, float32
        metadata: Словарь с метаданными и дополнительными атрибутами
    
    Поддерживаемые форматы:
        .txt/.xyz/.pts: Текстовые файлы (первые 3 столбца - X Y Z)
        .ptx: PTX формат (обрабатывается как текст)
        .ply: Через open3d
        .las/.laz: Через laspy с сохранением атрибутов
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    ext = path.suffix.lower()
    metadata: Dict[str, Any] = {
        'format': ext.lstrip('.'),
        'source_path': path.as_posix(),
        'filename': path.name
    }
    
    logger.info(f"Loading {ext} file: {path.name}")
    
    if ext in ['.txt', '.xyz', '.pts', '.ptx']:
        coords, meta = _load_text_format(path)
        metadata.update(meta)
    
    elif ext == '.ply':
        coords, meta = _load_ply_format(path)
        metadata.update(meta)
    
    elif ext in ['.las', '.laz']:
        coords, meta = _load_las_format(path)
        metadata.update(meta)
    
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")
    
    logger.info(f"Loaded {coords.shape[0]:,} points from {path.name}")
    return coords, metadata


def _load_text_format(path: Path) -> Tuple[np.ndarray, dict]:
    """Загрузка текстовых форматов (TXT, XYZ, PTS, PTX)"""
    try:
        # Пробуем загрузить как таблицу чисел
        arr = np.loadtxt(path.as_posix(), dtype=np.float32)
        
        # Обработка одномерного случая
        if arr.ndim == 1:
            if arr.size < 3:
                raise ValueError(f"В файле {path} меньше 3 чисел")
            arr = arr.reshape(-1, 3)
        
        if arr.shape[1] < 3:
            raise ValueError(f"В файле {path} меньше 3 столбцов (x y z)")
        
        # Основные координаты
        coords = arr[:, :3].astype(np.float32)
        
        # Дополнительные атрибуты
        metadata = {'columns': arr.shape[1]}
        if arr.shape[1] > 3:
            extras = arr[:, 3:].copy()
            metadata['extras'] = extras
            logger.debug(f"Found {arr.shape[1] - 3} extra columns")
        
        return coords, metadata
        
    except Exception as e:
        raise ValueError(f"Ошибка чтения текстового файла {path}: {e}")


def _load_ply_format(path: Path) -> Tuple[np.ndarray, dict]:
    """Загрузка PLY формата через open3d"""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Для формата .ply требуется пакет open3d\n"
            "Установите: pip install open3d"
        )
    
    try:
        pcd = o3d.io.read_point_cloud(path.as_posix())
        coords = np.asarray(pcd.points, dtype=np.float32)
        
        metadata = {}
        
        # Цвета (если есть)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)  # В диапазоне [0,1]
            metadata['rgb_float01'] = colors.astype(np.float32)
            logger.debug("Found RGB colors")
        
        # Нормали (если есть)
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            metadata['normals'] = normals.astype(np.float32)
            logger.debug("Found normals")
        
        return coords, metadata
        
    except Exception as e:
        raise ValueError(f"Ошибка чтения PLY файла {path}: {e}")


def _load_las_format(path: Path) -> Tuple[np.ndarray, dict]:
    """Загрузка LAS/LAZ формата через laspy"""
    try:
        import laspy
    except ImportError:
        raise ImportError(
            'Для .las/.laz требуется пакет laspy\n'
            'Установите: pip install "laspy[lazrs]"'
        )
    
    try:
        las = laspy.read(path.as_posix())
        coords = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        
        # Базовые метаданные LAS
        metadata = {
            'las_scales': np.array(las.header.scales, dtype=float),
            'las_offsets': np.array(las.header.offsets, dtype=float),
            'las_mins': np.array(las.header.mins, dtype=float),
            'las_maxs': np.array(las.header.maxs, dtype=float),
            'las_point_format': las.point_format.id,
            'las_point_count': las.header.point_count
        }
        
        # CRS информация
        try:
            metadata['las_crs'] = las.header.parse_crs()
            logger.debug(f"Found CRS: {metadata['las_crs']}")
        except Exception:
            metadata['las_crs'] = None
        
        # Популярные атрибуты
        dim_names = set(las.point_format.dimension_names)
        
        # RGB цвета
        if {'red', 'green', 'blue'} <= dim_names:
            metadata['rgb_u16'] = np.vstack([las.red, las.green, las.blue]).T.copy()
            logger.debug("Found RGB colors (16-bit)")
        
        # Интенсивность
        if 'intensity' in dim_names:
            metadata['intensity'] = las.intensity.copy()
            logger.debug("Found intensity values")
        
        # Классификация
        if 'classification' in dim_names:
            metadata['classification'] = las.classification.copy()
            unique_classes = np.unique(las.classification)
            metadata['unique_classes'] = unique_classes.tolist()
            logger.debug(f"Found classification: {len(unique_classes)} classes")
        
        # Return number
        if 'return_number' in dim_names:
            metadata['return_number'] = las.return_number.copy()
        
        # Number of returns
        if 'number_of_returns' in dim_names:
            metadata['number_of_returns'] = las.number_of_returns.copy()
        
        # GPS time
        if 'gps_time' in dim_names:
            metadata['gps_time'] = las.gps_time.copy()
            logger.debug("Found GPS time")
        
        return coords, metadata
        
    except Exception as e:
        raise ValueError(f"Ошибка чтения LAS/LAZ файла {path}: {e}")


def validate_point_cloud(points: NDArray[np.floating], 
                        min_points: int = 10,
                        max_points: Optional[int] = None) -> None:
    """
    Валидация загруженного облака точек
    
    Args:
        points: Массив координат
        min_points: Минимальное количество точек
        max_points: Максимальное количество точек (опционально)
    
    Raises:
        ValueError: Если данные не соответствуют требованиям
    """
    if not isinstance(points, np.ndarray):
        raise TypeError("Expected numpy.ndarray")
    
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")
    
    n = points.shape[0]
    
    if n < min_points:
        raise ValueError(f"Too few points: {n} < {min_points}")
    
    if max_points and n > max_points:
        raise ValueError(f"Too many points: {n} > {max_points}")
    
    # Геометрические проверки делаем только по XYZ-координатам
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected array of shape (N, >=3), got {points.shape}")
    xyz = np.asarray(points[:, :3], dtype=np.float64)

    # Проверка на NaN/Inf только по XYZ
    if not np.isfinite(xyz).all():
        n_invalid = (~np.isfinite(xyz)).any(axis=1).sum()
        raise ValueError(f"Found {n_invalid} points with NaN or Inf coordinates")
   
    # NumPy 2.0+: используем функциональный стиль np.ptp на XYZ
    bbox = np.ptp(xyz, axis=0)  # эквивалент: xyz.max(axis=0) - xyz.min(axis=0)
    EPS = 1e-12
    if np.any(bbox <= EPS):
        zero_dims = np.where(bbox <= EPS)[0]
        dim_names = ['x', 'y', 'z']
        zero_names = [dim_names[i] for i in zero_dims]
        raise ValueError(f"Point cloud is degenerate along {zero_names} (ptp={bbox}, eps={EPS})")

def subsample_points(points: np.ndarray,
                     max_points: int,
                     method: str = 'random',
                     random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подвыборка облака точек
    
    Args:
        points: Исходные точки
        max_points: Максимальное количество точек
        method: Метод подвыборки ('random', 'uniform', 'voxel')
        random_seed: Seed для воспроизводимости
    
    Returns:
        (subsampled_points, indices) - подвыборка и индексы выбранных точек
    """
    n = points.shape[0]
    
    if n <= max_points:
        return points, np.arange(n)
    
    if method == 'random':
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(n, size=max_points, replace=False)
        indices.sort()
        return points[indices], indices
    
    elif method == 'uniform':
        # Равномерная подвыборка
        step = n / max_points
        indices = np.arange(0, n, step).astype(int)[:max_points]
        return points[indices], indices
    
    elif method == 'voxel':
        # Воксельная подвыборка (требует open3d)
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Вычисляем размер вокселя
            bbox_diag = np.linalg.norm(points.ptp(points[:, :3], axis=0))
            voxel_size = bbox_diag / (max_points ** (1/3))
            
            downsampled = pcd.voxel_down_sample(voxel_size)
            subsampled = np.asarray(downsampled.points)
            
            # Находим ближайшие исходные точки
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(points)
            _, indices = nn.kneighbors(subsampled)
            indices = indices.flatten()
            
            return points[indices], indices
            
        except ImportError:
            logger.warning("open3d not available, falling back to random sampling")
            return subsample_points(points, max_points, 'random', random_seed)
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")


# =====================================================
# bvh_partition/io/__init__.py
# =====================================================
"""
Модуль ввода-вывода для BVH разбиения
"""
from .loaders import (
    load_point_cloud,
    validate_point_cloud,
    subsample_points
)

__all__ = [
    'load_point_cloud',
    'validate_point_cloud',
    'subsample_points'
]
