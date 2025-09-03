from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class AABB:
    """
    Axis-Aligned Bounding Box в целочисленной сетке
    
    Attributes:
        min_corner: Минимальные координаты в сетке (включительно)
        max_corner: Максимальные координаты в сетке (исключительно)
        closed: Флаги включения границ (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    min_corner: np.ndarray  # shape: (3,), dtype: int64
    max_corner: np.ndarray  # shape: (3,), dtype: int64
    closed: Tuple[bool, bool, bool, bool, bool, bool]
    
    def __post_init__(self):
        """Валидация данных после инициализации"""
        self.min_corner = np.asarray(self.min_corner, dtype=np.int64)
        self.max_corner = np.asarray(self.max_corner, dtype=np.int64)
        
        if self.min_corner.shape != (3,) or self.max_corner.shape != (3,):
            raise ValueError("AABB corners must be 3D vectors")
        
        if np.any(self.min_corner > self.max_corner):
            raise ValueError("min_corner must be <= max_corner")
    
    def volume(self) -> float:
        """Объём параллелепипеда в ячейках сетки"""
        return float(np.prod(self.max_corner - self.min_corner))
    
    def surface_area(self) -> float:
        """Площадь поверхности параллелепипеда в единицах сетки²"""
        dx, dy, dz = self.max_corner - self.min_corner
        return float(2 * (dx * dy + dx * dz + dy * dz))
    
    def dimensions(self) -> np.ndarray:
        """Размеры по осям в ячейках сетки"""
        return self.max_corner - self.min_corner
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Проверка принадлежности точки боксу с учётом флагов closed"""
        for i in range(3):
            min_val, max_val = self.min_corner[i], self.max_corner[i]
            min_closed = self.closed[2 * i]
            max_closed = self.closed[2 * i + 1]
            
            if min_closed:
                if point[i] < min_val:
                    return False
            else:
                if point[i] <= min_val:
                    return False
            
            if max_closed:
                if point[i] > max_val:
                    return False
            else:
                if point[i] >= max_val:
                    return False
        
        return True
    
    def to_world(self, scale_m: float, offset: Optional[np.ndarray] = None) -> dict:
        """
        Преобразование в мировые координаты
        
        Args:
            scale_m: Масштаб сетки (метры на ячейку)
            offset: Смещение начала координат (опционально)
        
        Returns:
            Словарь с ключами 'min' и 'max' в метрах
        """
        min_world = self.min_corner.astype(np.float64) * scale_m
        max_world = self.max_corner.astype(np.float64) * scale_m
        
        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64)
            min_world += offset
            max_world += offset
        
        return {
            'min': min_world.tolist(),
            'max': max_world.tolist()
        }


@dataclass
class Node:
    """
    Узел BVH дерева
    
    Attributes:
        aabb: Ограничивающий параллелепипед
        indices: Индексы точек в этом узле
        level: Уровень в дереве (0 для корня)
        children: Пара дочерних узлов (None для листьев)
    """
    aabb: AABB
    indices: np.ndarray  # shape: (n,), dtype: int64
    level: int
    children: Optional[Tuple['Node', 'Node']] = None
    
    def __post_init__(self):
        """Валидация и преобразование типов"""
        self.indices = np.asarray(self.indices, dtype=np.int64)
        
        if self.level < 0:
            raise ValueError("Level must be non-negative")
    
    def is_leaf(self) -> bool:
        """Проверка, является ли узел листом"""
        return self.children is None
    
    def point_count(self) -> int:
        """Количество точек в узле"""
        return int(self.indices.size)
    
    def depth(self) -> int:
        """Глубина поддерева с корнем в данном узле"""
        if self.is_leaf():
            return 0
        left_depth = self.children[0].depth()
        right_depth = self.children[1].depth()
        return 1 + max(left_depth, right_depth)
    
    def leaf_count(self) -> int:
        """Количество листьев в поддереве"""
        if self.is_leaf():
            return 1
        return self.children[0].leaf_count() + self.children[1].leaf_count()
    
    def node_count(self) -> int:
        """Общее количество узлов в поддереве"""
        if self.is_leaf():
            return 1
        return 1 + self.children[0].node_count() + self.children[1].node_count()
    
    def iter_leaves(self):
        """Итератор по листовым узлам"""
        stack = [self]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                yield node
            else:
                stack.extend(node.children)
    
    def get_stats(self) -> dict:
        """Статистика поддерева"""
        leaf_sizes = [leaf.point_count() for leaf in self.iter_leaves()]
        return {
            'depth': self.depth(),
            'node_count': self.node_count(),
            'leaf_count': self.leaf_count(),
            'total_points': sum(leaf_sizes),
            'min_leaf_size': min(leaf_sizes) if leaf_sizes else 0,
            'max_leaf_size': max(leaf_sizes) if leaf_sizes else 0,
            'median_leaf_size': int(np.median(leaf_sizes)) if leaf_sizes else 0
        }