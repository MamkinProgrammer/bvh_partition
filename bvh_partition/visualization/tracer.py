"""
Трассировщик для сбора данных визуализации процесса построения
"""
import json
import csv
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TraceRecorder:
    """
    Сборщик артефактов для визуализации процесса построения BVH
    
    Записывает ключевые этапы:
    1. Входные точки (подвыборка)
    2. Линии сетки квантизации
    3. Выбранный узел для анализа плотности
    4. Процесс поиска оптимального разреза
    5. Операции слияния
    6. Финальные блоки
    """
    max_points_sample: int = 8000
    grid_lines_target: int = 8
    data: dict = field(default_factory=dict)
    
    # Флаги для предотвращения дублирования
    _have_input: bool = False
    _have_grid: bool = False
    _have_density_box: bool = False
    _have_split: bool = False
    _have_merge: bool = False
    _stats_file: Optional[Any] = None
    _stats_writer: Optional[Any] = None
    _stats_header_written: bool = False
    
    def __post_init__(self):
        """Инициализация структуры данных"""
        self.data = {
            'metadata': {
                'version': '1.0',
                'description': 'BVH construction trace for visualization'
            },
            'steps': {}
        }
    
    def record_input_points(self, points_xy: np.ndarray) -> None:
        """
        Запись подвыборки входных точек (только XY проекция)
        
        Args:
            points_xy: 2D координаты точек (N x 2)
        """
        if self._have_input:
            return
        
        n = min(len(points_xy), self.max_points_sample)
        
        # Детерминированная подвыборка для воспроизводимости
        rng = np.random.default_rng(42)
        if len(points_xy) > n:
            indices = rng.choice(len(points_xy), size=n, replace=False)
        else:
            indices = np.arange(len(points_xy))
        
        sample = points_xy[indices]
        
        self.data['input_points'] = [
            {'x': float(x), 'y': float(y)} 
            for x, y in sample.tolist()
        ]
        
        self._have_input = True
        logger.debug(f"Recorded {n} input points for visualization")
    
    def record_grid_lines(self, 
                         min_corner: np.ndarray,
                         max_corner: np.ndarray,
                         scale_m: float) -> None:
        """
        Запись линий сетки для визуализации квантизации
        
        Args:
            min_corner: Минимальные координаты сетки
            max_corner: Максимальные координаты сетки
            scale_m: Масштаб сетки (м/ячейку)
        """
        if self._have_grid:
            return
        
        def generate_lines_1d(lo: int, hi: int) -> List[float]:
            """Генерация позиций линий сетки"""
            count = max(2, int(self.grid_lines_target))
            indices = np.linspace(int(lo), int(hi), num=count, endpoint=True)
            return [float(idx * scale_m) for idx in indices]
        
        self.data['step_1_quantization'] = {
            'grid_lines_v': generate_lines_1d(min_corner[0], max_corner[0]),
            'grid_lines_h': generate_lines_1d(min_corner[1], max_corner[1]),
            'scale_m': float(scale_m),
            'grid_bounds': {
                'min': min_corner.tolist(),
                'max': max_corner.tolist()
            }
        }
        
        self._have_grid = True
        logger.debug(f"Recorded grid with scale {scale_m:.6f} m/cell")
    
    def record_density_box(self, 
                          aabb: Any,
                          scale_m: float) -> None:
        """
        Запись выбранного блока для анализа плотности
        
        Args:
            aabb: AABB узла
            scale_m: Масштаб сетки
        """
        if self._have_density_box:
            return
        
        min_world = aabb.min_corner.astype(float) * float(scale_m)
        max_world = aabb.max_corner.astype(float) * float(scale_m)
        
        self.data['step_2_density_analysis'] = {
            'highlighted_box': {
                'min': [float(min_world[0]), float(min_world[1])],
                'max': [float(max_world[0]), float(max_world[1])]
            }
        }
        
        self._have_density_box = True
        logger.debug("Recorded density analysis box")
    
    def record_split(self,
                    parent_aabb: Any,
                    axis: int,
                    cut_index: int,
                    scale_m: float,
                    candidate_splits: Optional[List[float]] = None,
                    chosen_split_pos: Optional[float] = None) -> None:
        """
        Запись процесса поиска оптимального разреза
        
        Args:
            parent_aabb: AABB родительского узла
            axis: Ось разреза (0=X, 1=Y, 2=Z)
            cut_index: Позиция разреза в сетке
            scale_m: Масштаб сетки
            candidate_splits: Рассмотренные кандидаты (в метрах)
            chosen_split_pos: Выбранная позиция (в метрах)
        """
        if self._have_split:
            return
        
        # Границы родителя в мировых координатах
        min_world = parent_aabb.min_corner.astype(float) * float(scale_m)
        max_world = parent_aabb.max_corner.astype(float) * float(scale_m)
        
        parent_box = {
            'min': [float(min_world[0]), float(min_world[1])],
            'max': [float(max_world[0]), float(max_world[1])]
        }
        
        # Имя оси
        axis_names = ['X', 'Y', 'Z']
        axis_name = axis_names[axis] if 0 <= axis < 3 else str(axis)
        
        # Позиция разреза
        if chosen_split_pos is None:
            chosen_split_pos = float(cut_index) * float(scale_m)
        
        # Кандидаты (если не переданы - хотя бы выбранный)
        if not candidate_splits:
            candidate_splits = [chosen_split_pos]
        
        self.data['step_3_split_search'] = {
            'parent_box': parent_box,
            'split_axis': axis_name,
            'candidate_splits': [float(x) for x in candidate_splits],
            'chosen_split_pos': float(chosen_split_pos)
        }
        
        self._have_split = True
        logger.debug(f"Recorded split along {axis_name} axis at {chosen_split_pos:.3f}")
    
    def record_merge(self,
                    boxes_to_merge: List[Any],
                    merged_box: Any,
                    scale_m: float) -> None:
        """
        Запись операции слияния листьев
        
        Args:
            boxes_to_merge: Список AABB сливаемых узлов
            merged_box: Результирующий AABB
            scale_m: Масштаб сетки
        """
        if self._have_merge:
            return
        
        def aabb_to_world(aabb):
            """Конвертация AABB в мировые координаты"""
            min_w = aabb.min_corner.astype(float) * float(scale_m)
            max_w = aabb.max_corner.astype(float) * float(scale_m)
            return {
                'min': [float(min_w[0]), float(min_w[1])],
                'max': [float(max_w[0]), float(max_w[1])]
            }
        
        self.data['step_4_merging'] = {
            'boxes_to_merge': [aabb_to_world(b) for b in boxes_to_merge],
            'merged_box': aabb_to_world(merged_box)
        }
        
        self._have_merge = True
        logger.debug(f"Recorded merge of {len(boxes_to_merge)} boxes")
    
    def record_final_boxes(self,
                          leaves_iter,
                          scale_m: float) -> None:
        """
        Запись финальных блоков (листьев дерева)
        
        Args:
            leaves_iter: Итератор по листовым узлам
            scale_m: Масштаб сетки
        """
        final_boxes = []
        
        for leaf in leaves_iter:
            aabb = leaf.aabb
            min_world = aabb.min_corner.astype(float) * float(scale_m)
            max_world = aabb.max_corner.astype(float) * float(scale_m)
            
            final_boxes.append({
                'min': [float(min_world[0]), float(min_world[1]), float(min_world[2])],
                'max': [float(max_world[0]), float(max_world[1]), float(max_world[2])],
                'count': int(leaf.point_count()),
                'level': int(leaf.level)
            })
        
        self.data['step_5_final_output'] = {
            'final_boxes': final_boxes,
            'total_leaves': len(final_boxes),
            'total_points': sum(b['count'] for b in final_boxes)
        }
        
        logger.debug(f"Recorded {len(final_boxes)} final blocks")
    
    def add_custom_data(self, key: str, value: Any) -> None:
        """
        Добавление пользовательских данных в трассировку
        
        Args:
            key: Ключ для данных
            value: Значение (должно быть JSON-сериализуемым)
        """
        if 'custom' not in self.data:
            self.data['custom'] = {}
        
        self.data['custom'][key] = value
        logger.debug(f"Added custom trace data: {key}")
    
    def start_stats_recording(self, path: Path):
        """Открывает CSV-файл для записи статистики разбиений."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stats_file = open(path, 'w', newline='', encoding='utf-8')
            self._stats_writer = csv.writer(self._stats_file)
            logger.info(f"Split statistics recording enabled, saving to {path}")
        except IOError as e:
            logger.error(f"Failed to open stats file {path}: {e}")
            self._stats_file = None
            self._stats_writer = None

    def record_split_decision(self, data: dict):
        """Записывает одну строку данных о принятом решении в CSV."""
        if not self._stats_writer:
            return
            
        try:
            if not self._stats_header_written:
                self._stats_writer.writerow(data.keys())
                self._stats_header_written = True
            self._stats_writer.writerow(data.values())
        except Exception as e:
            logger.error(f"Error recording split decision stats: {e}")

    def close(self):
        """Закрывает все открытые файлы (JSON и CSV)."""
        if self._stats_file:
            self._stats_file.close()
            logger.debug("Stats CSV file closed.")
    
    def dump(self, path: Path) -> None:
        """
        Сохранение трассировки в JSON файл
        
        Args:
            path: Путь к выходному файлу
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Trace saved to {path}")
    
    def get_summary(self) -> dict:
        """
        Получение краткой сводки по трассировке
        
        Returns:
            Словарь со статистикой
        """
        summary = {
            'has_input_points': self._have_input,
            'has_grid': self._have_grid,
            'has_density_box': self._have_density_box,
            'has_split': self._have_split,
            'has_merge': self._have_merge
        }
        
        if 'input_points' in self.data:
            summary['n_input_points'] = len(self.data['input_points'])
        
        if 'step_5_final_output' in self.data:
            final = self.data['step_5_final_output']
            summary['n_final_boxes'] = final.get('total_leaves', 0)
            summary['n_total_points'] = final.get('total_points', 0)
        
        return summary


# =====================================================
# bvh_partition/visualization/__init__.py
# =====================================================
"""
Модуль визуализации для BVH разбиения
"""
from .tracer import TraceRecorder

__all__ = ['TraceRecorder']
