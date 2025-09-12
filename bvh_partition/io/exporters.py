import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import asdict
import logging

from ..core.structures import Node

logger = logging.getLogger(__name__)


def export_leaves(root: Node,
                 points: np.ndarray,
                 output_dir: str,
                 formats: List[str],
                 metadata: Optional[dict] = None,
                 scale_m: Optional[float] = None) -> None:
    """
    Экспорт листовых узлов в указанные форматы
    
    Args:
        root: Корневой узел дерева
        points: Исходные точки
        output_dir: Выходная директория
        formats: Список форматов ['json', 'las', 'ply', 'xyz', 'txt']
        metadata: Метаданные из входного файла
        scale_m: Масштаб сетки (для JSON)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        if fmt == 'none':
            continue
        
        logger.info(f"Exporting to {fmt.upper()} format...")
        
        if fmt == 'json':
            export_leaves_json(root, out_path / 'blocks.json', scale_m)
            
        elif fmt == 'las':
            export_leaves_las(root, points, out_path / 'blocks_las', metadata)
            
        elif fmt == 'ply':
            export_leaves_ply(root, points, out_path / 'blocks_ply', metadata)
            
        elif fmt in ['xyz', 'txt', 'pts']:
            export_leaves_text(root, points, out_path / f'blocks_{fmt}', 
                              metadata, extension=fmt)
            
        elif fmt == 'same':
            # Экспорт в исходном формате
            original_fmt = metadata.get('format', 'xyz') if metadata else 'xyz'
            export_same_format(root, points, out_path / 'blocks_same', 
                             metadata, original_fmt)
        else:
            logger.warning(f"Unknown export format: {fmt}")


def export_leaves_json(root: Node, 
                       output_file: Path,
                       scale_m: Optional[float] = None) -> None:
    """
    Экспорт листовых узлов в JSON
    
    Формат:
    [
        {
            "min": [x, y, z],      # В единицах сетки или метрах
            "max": [x, y, z],
            "closed": [bool, ...], # Флаги включения границ
            "count": int,          # Количество точек
            "level": int           # Уровень в дереве
        },
        ...
    ]
    """
    leaves_data = []
    
    for leaf in root.iter_leaves():
        if scale_m:
            # Конвертация в мировые координаты
            box_world = leaf.aabb.to_world(scale_m)
            leaf_dict = {
                'min': box_world['min'],
                'max': box_world['max'],
                'closed': list(leaf.aabb.closed),
                'count': leaf.point_count(),
                'level': leaf.level
            }
        else:
            # В единицах сетки
            leaf_dict = {
                'min': leaf.aabb.min_corner.tolist(),
                'max': leaf.aabb.max_corner.tolist(),
                'closed': list(leaf.aabb.closed),
                'count': leaf.point_count(),
                'level': leaf.level
            }
        
        leaves_data.append(leaf_dict)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(leaves_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Exported {len(leaves_data)} leaves to {output_file}")


def export_leaves_text(root: Node,
                      points: np.ndarray,
                      output_dir: Path,
                      metadata: Optional[dict] = None,
                      extension: str = 'xyz') -> None:
    """
    Экспорт листовых узлов в текстовые форматы
    
    Args:
        root: Корневой узел
        points: Массив точек
        output_dir: Директория для файлов
        metadata: Метаданные с дополнительными колонками
        extension: Расширение файлов ('xyz', 'txt', 'pts')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Дополнительные атрибуты из входного файла
    extras = metadata.get('extras', None) if metadata else None
    
    for i, leaf in enumerate(root.iter_leaves()):
        indices = leaf.indices
        leaf_points = points[indices]
        
        # Добавляем дополнительные колонки если есть
        if extras is not None and extras.shape[0] >= points.shape[0]:
            output_data = np.hstack([leaf_points, extras[indices]])
        else:
            output_data = leaf_points
        
        output_file = output_dir / f'block_{i:05d}.{extension}'
        np.savetxt(output_file.as_posix(), output_data, fmt='%.6f')
    
    n_leaves = i + 1 if 'i' in locals() else 0
    logger.info(f"Exported {n_leaves} blocks to {output_dir}")


def export_leaves_las(root: Node,
                     points: np.ndarray,
                     output_dir: Path,
                     metadata: Optional[dict] = None,
                     global_scale: Optional[Tuple[float, float, float]] = None,
                     use_auto_offset: bool = True) -> None:
    """
    Экспорт листовых узлов в LAS формат
    
    Сохраняет scales/offsets/CRS из исходного файла если доступны
    """
    try:
        import laspy
    except ImportError:
        raise ImportError('Для экспорта в LAS требуется пакет laspy')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Настройка scale/offset
    if metadata and 'las_scales' in metadata and 'las_offsets' in metadata:
        scales = np.array(metadata['las_scales'], dtype=np.float64)
        offsets = np.array(metadata['las_offsets'], dtype=np.float64)
        crs = metadata.get('las_crs', None)
    else:
        scales = np.array(global_scale if global_scale else [1e-3, 1e-3, 1e-3])
        offsets = points.min(axis=0) if use_auto_offset else np.zeros(3)
        crs = None
        
        # Проверка на переполнение int32
        span = points.max(axis=0) - offsets
        max_int = (span / scales).max()
        if max_int > 2_000_000_000:
            factor = max_int / 1_000_000_000.0
            scales *= factor
            logger.warning(f"Adjusted LAS scales by factor {factor:.2f} to prevent overflow")
    
    # Дополнительные атрибуты
    rgb_u16 = metadata.get('rgb_u16') if metadata else None
    intensity = metadata.get('intensity') if metadata else None
    classification = metadata.get('classification') if metadata else None
    return_number = metadata.get('return_number') if metadata else None
    
    for i, leaf in enumerate(root.iter_leaves()):
        indices = leaf.indices
        leaf_points = points[indices]
        
        # Создание LAS файла
        header = laspy.LasHeader(point_format=3, version='1.2')
        header.scales = scales
        header.offsets = offsets
        
        if crs is not None:
            try:
                header.add_crs(crs)
            except Exception:
                pass
        
        las = laspy.LasData(header)
        las.x = leaf_points[:, 0]
        las.y = leaf_points[:, 1]
        las.z = leaf_points[:, 2]
        
        # Копирование атрибутов
        if rgb_u16 is not None and rgb_u16.shape[0] >= points.shape[0]:
            sub_rgb = rgb_u16[indices]
            try:
                las.red = sub_rgb[:, 0]
                las.green = sub_rgb[:, 1]
                las.blue = sub_rgb[:, 2]
            except Exception:
                pass
        
        if intensity is not None and intensity.shape[0] >= points.shape[0]:
            try:
                las.intensity = intensity[indices]
            except Exception:
                pass
        
        if classification is not None and classification.shape[0] >= points.shape[0]:
            try:
                las.classification = classification[indices]
            except Exception:
                pass
        
        if return_number is not None and return_number.shape[0] >= points.shape[0]:
            try:
                las.return_number = return_number[indices]
            except Exception:
                pass
        
        output_file = output_dir / f'block_{i:05d}.las'
        las.write(output_file.as_posix())
    
    n_leaves = i + 1 if 'i' in locals() else 0
    logger.info(f"Exported {n_leaves} LAS blocks to {output_dir}")


def export_leaves_ply(root: Node,
                     points: np.ndarray,
                     output_dir: Path,
                     metadata: Optional[dict] = None) -> None:
    """
    Экспорт листовых узлов в PLY формат
    
    Поддерживает цвета из метаданных (rgb_float01 или rgb_u16)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Извлечение цветов
    rgb01 = None
    if metadata:
        if 'rgb_float01' in metadata:
            rgb01 = metadata['rgb_float01']
        elif 'rgb_u16' in metadata:
            rgb01 = (metadata['rgb_u16'].astype(np.float32) / 65535.0).clip(0, 1)
    
    # Попытка использовать open3d
    try:
        import open3d as o3d
        use_o3d = True
    except ImportError:
        use_o3d = False
        logger.warning("open3d not available, using ASCII PLY export")
    
    for i, leaf in enumerate(root.iter_leaves()):
        indices = leaf.indices
        leaf_points = points[indices]
        
        if use_o3d:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(leaf_points.astype(np.float64))
            
            if rgb01 is not None and rgb01.shape[0] >= points.shape[0]:
                pcd.colors = o3d.utility.Vector3dVector(rgb01[indices].astype(np.float64))
            
            output_file = output_dir / f'block_{i:05d}.ply'
            o3d.io.write_point_cloud(output_file.as_posix(), pcd, write_ascii=True)
        
        else:
            # Минимальный ASCII PLY
            output_file = output_dir / f'block_{i:05d}.ply'
            _write_ascii_ply(output_file, leaf_points, 
                           rgb01[indices] if rgb01 is not None else None)
    
    n_leaves = i + 1 if 'i' in locals() else 0
    logger.info(f"Exported {n_leaves} PLY blocks to {output_dir}")


def _write_ascii_ply(output_file: Path,
                    points: np.ndarray,
                    colors: Optional[np.ndarray] = None) -> None:
    """Запись минимального ASCII PLY файла"""
    with open(output_file, 'w', encoding='ascii') as f:
        has_color = colors is not None and colors.shape[0] == points.shape[0]
        
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # Data
        for j in range(points.shape[0]):
            if has_color:
                r, g, b = (colors[j] * 255.0 + 0.5).astype(np.uint8)
                f.write(f"{points[j,0]} {points[j,1]} {points[j,2]} {r} {g} {b}\n")
            else:
                f.write(f"{points[j,0]} {points[j,1]} {points[j,2]}\n")


def export_same_format(root: Node,
                      points: np.ndarray,
                      output_dir: Path,
                      metadata: Optional[dict],
                      original_format: str) -> None:
    """
    Экспорт в том же формате что и входной файл
    
    Args:
        root: Корневой узел
        points: Массив точек
        output_dir: Выходная директория
        metadata: Метаданные входного файла
        original_format: Исходный формат
    """
    fmt = original_format.lower()
    
    if fmt in ['las', 'laz']:
        export_leaves_las(root, points, output_dir, metadata)
    
    elif fmt == 'ply':
        export_leaves_ply(root, points, output_dir, metadata)
    
    elif fmt in ['txt', 'xyz', 'pts']:
        export_leaves_text(root, points, output_dir, metadata, extension=fmt)
    
    elif fmt == 'ptx':
        # PTX экспортируем как XYZ (без сетки сканера)
        logger.warning("PTX export not fully supported, using XYZ format")
        export_leaves_text(root, points, output_dir, metadata, extension='xyz')
    
    else:
        logger.warning(f"Cannot export to original format '{fmt}', using XYZ")
        export_leaves_text(root, points, output_dir, metadata, extension='xyz')


def export_statistics(root: Node,
                     points: np.ndarray,
                     output_file: Path,
                     build_time: Optional[float] = None,
                     peak_memory_mb: Optional[float] = None,
                     cpu_time_sec: Optional[float] = None,
                     config: Optional[Any] = None) -> None:
    """
    Экспорт статистики построения дерева
    
    Args:
        root: Корневой узел
        points: Массив точек
        output_file: Путь к выходному JSON файлу
        build_time: Время построения (секунды)
        config: Конфигурация построителя
    """
    stats = root.get_stats()
    
    # Размеры листьев
    leaf_sizes = [leaf.point_count() for leaf in root.iter_leaves()]
    
    result = {
        'input': {
            'total_points': points.shape[0],
            'bbox_min': points.min(axis=0).tolist(),
            'bbox_max': points.max(axis=0).tolist()
        },
        'tree': {
            'depth': stats['depth'],
            'total_nodes': stats['node_count'],
            'leaf_nodes': stats['leaf_count'],
            'internal_nodes': stats['node_count'] - stats['leaf_count']
        },
        'leaves': {
            'min_size': min(leaf_sizes) if leaf_sizes else 0,
            'max_size': max(leaf_sizes) if leaf_sizes else 0,
            'mean_size': float(np.mean(leaf_sizes)) if leaf_sizes else 0,
            'median_size': float(np.median(leaf_sizes)) if leaf_sizes else 0,
            'std_size': float(np.std(leaf_sizes)) if leaf_sizes else 0
        }
    }
    
    if build_time is not None:
        result['performance'] = {
            'build_time_wall_sec': build_time,
            'build_time_cpu_sec': cpu_time_sec,
            'peak_memory_mb': peak_memory_mb,
            'points_per_sec_wall': points.shape[0] / build_time if build_time > 0 else 0,
            'points_per_sec_cpu': points.shape[0] / cpu_time_sec if cpu_time_sec > 0 else 0
        }
    
    if config is not None:
        result['config'] = asdict(config)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Exported statistics to {output_file}")
