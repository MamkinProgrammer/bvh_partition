"""
Метрики для выбора оптимального разбиения
"""
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

from ..config import MIN_CHILD_FRACTION, VOLUME_EPSILON


@dataclass
class SplitCandidate:
    """Кандидат на разрез"""
    axis: int  # 0=X, 1=Y, 2=Z
    position: int  # Позиция в сетке
    cost: float  # Метрика стоимости
    left_count: int  # Точек слева
    right_count: int  # Точек справа
    method: str  # 'sah' или 'gmeans'
    
    def is_valid(self, min_points: int) -> bool:
        """Проверка валидности разреза"""
        return self.left_count >= min_points and self.right_count >= min_points


class MetricsCalculator:
    """Вычислитель метрик для выбора разреза"""
    
    def __init__(self, config):
        self.config = config
        self.scale_m = 1.0  # Будет установлен извне
    
    def compute_sah_cost(self,
                         n_left: int,
                         n_right: int,
                         surface_left: float,
                         surface_right: float,
                         surface_parent: float) -> float:
        """
        Surface Area Heuristic (SAH)
        
        Args:
            n_left, n_right: Количество точек слева/справа
            surface_left, surface_right: Площади поверхности дочерних узлов
            surface_parent: Площадь поверхности родителя
        
        Returns:
            SAH стоимость разбиения
        """
        if surface_parent <= 0:
            return float('inf')
        
        return (surface_left * n_left + surface_right * n_right) / surface_parent
    
    def compute_density_gap(self,
                           n_left: int,
                           n_right: int,
                           area_left: float,
                           area_right: float,
                           target_density: float) -> float:
        """
        Метрика разницы плотностей
        
        Args:
            n_left, n_right: Количество точек
            area_left, area_right: Площади проекций (м²)
            target_density: Целевая плотность (точек/м²)
        
        Returns:
            Нормализованная разница плотностей
        """
        if target_density <= 0:
            return 0.0
        
        density_left = n_left / max(area_left, VOLUME_EPSILON)
        density_right = n_right / max(area_right, VOLUME_EPSILON)
        
        return abs(density_left - density_right) / target_density
    
    def compute_height_contrast(self,
                               z_coords_left: np.ndarray,
                               z_coords_right: np.ndarray,
                               z_range: float) -> float:
        """
        Метрика контраста высот
        
        Args:
            z_coords_left, z_coords_right: Z-координаты точек (в метрах)
            z_range: Диапазон высот родителя
        
        Returns:
            Нормализованный контраст средних высот
        """
        if z_range <= 0 or len(z_coords_left) == 0 or len(z_coords_right) == 0:
            return 0.0
        
        mean_left = np.mean(z_coords_left)
        mean_right = np.mean(z_coords_right)
        
        return abs(mean_left - mean_right) / z_range
    
    def compute_combined_cost(self,
                             sah_cost: float,
                             density_gap: float = 0.0,
                             height_contrast: float = 0.0) -> float:
        """
        Комбинированная метрика
        
        Cost = SAH - λ_D * DensityGap - λ_H * HeightContrast
        """
        cost = sah_cost
        cost -= self.config.lambda_density * density_gap
        cost -= self.config.lambda_height * height_contrast
        return cost
    
    def evaluate_split_quality(self,
                              parent_volume: float,
                              parent_count: int,
                              left_count: int,
                              right_count: int) -> float:
        """
        Оценка качества разбиения (0 = плохо, 1 = идеально)
        
        Учитывает:
        - Баланс разбиения
        - Соответствие динамическому порогу
        """
        if parent_count == 0:
            return 0.0
        
        # Баланс (0.5 = идеально сбалансировано)
        balance = min(left_count, right_count) / parent_count
        
        # Соответствие динамическому порогу
        dynamic_max = self._compute_dynamic_threshold(parent_volume)
        size_ratio = max(left_count, right_count) / dynamic_max
        size_score = 1.0 - abs(1.0 - size_ratio) if size_ratio <= 1.0 else 1.0 / size_ratio
        
        return balance * 0.4 + size_score * 0.6
    
    def _compute_dynamic_threshold(self, volume: float) -> int:
        """Вычисление динамического порога размера узла"""
        threshold = self.config.rho_target * (volume ** self.config.gamma)
        return int(np.clip(threshold,
                          self.config.min_points_per_leaf,
                          self.config.max_points_per_leaf))
    
    def should_split(self,
                     node_count: int,
                     node_volume: float,
                     node_dimensions: np.ndarray) -> bool:
        """
        Проверка необходимости разбиения узла
        
        Args:
            node_count: Количество точек в узле
            node_volume: Объём узла в ячейках
            node_dimensions: Размеры узла по осям
        
        Returns:
            True если узел нужно разбивать
        """
        # Проверка минимальной высоты
        if np.any(node_dimensions < self.config.min_height):
            return False
        
        # Проверка динамического порога
        dynamic_max = self._compute_dynamic_threshold(node_volume)
        
        return node_count > dynamic_max
    
    def can_merge_leaves(self,
                        left_count: int,
                        right_count: int,
                        parent_volume: float) -> bool:
        """
        Проверка возможности слияния двух листьев
        
        Args:
            left_count, right_count: Количество точек в листьях
            parent_volume: Объём родительского узла
        
        Returns:
            True если листья можно объединить
        """
        if not self.config.merge_enabled:
            return False
        
        total_count = left_count + right_count
        dynamic_max = self._compute_dynamic_threshold(parent_volume)
        
        return total_count <= dynamic_max


class GmeansEvaluator:
    """Детектор мультимодальности на основе G-means"""
    
    def __init__(self, config, random_state: Optional[int] = None):
        self.config = config
        self.random_state = random_state or config.random_seed
        self.rng = np.random.default_rng(self.random_state)
        
        # Опциональные зависимости
        self._anderson = None
        self._kmeans = None
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Проверка наличия опциональных библиотек"""
        try:
            from scipy.stats import anderson
            self._anderson = anderson
        except ImportError:
            pass
        
        try:
            from sklearn.cluster import KMeans
            self._kmeans = KMeans
        except ImportError:
            pass
    
    def evaluate(self,
                coords: np.ndarray,
                indices: np.ndarray,
                sample_size: int = 200_000) -> Tuple[bool, int, float, Optional[float]]:
        """
        Поиск мультимодальности вдоль осей
        
        Args:
            coords: Координаты всех точек в сетке
            indices: Индексы точек текущего узла
            sample_size: Размер выборки для анализа
        
        Returns:
            (trigger, axis, cut_position, mid_world_position)
            trigger: True если найдена мультимодальность
            axis: Ось с максимальным разделением (0, 1, 2)
            cut_position: Позиция разреза в сетке
            mid_world_position: Позиция разреза в метрах
        """
        if not self.config.gmeans_enabled:
            return False, -1, -1, None
        
        n_sample = min(sample_size, indices.size)
        if n_sample <= 1:
            return False, -1, -1, None
        
        sample_indices = self.rng.choice(indices, n_sample, replace=False)
        
        scale = getattr(self.config, "scale_runtime", None)
        if scale is None:
            scale = getattr(self.config, "scale_fixed", None)
        if scale is None:
            scale = 1.0
        best_separation = -np.inf
        best_axis = -1
        best_cut = -1
        best_mid_world = None
        
        for axis in range(3):
            data = coords[sample_indices, axis].astype(np.float64)
            
            # Пропускаем константные данные
            if np.all(data == data[0]):
                continue
            
            # Кластеризация на 2 группы
            centers, sigmas = self._find_clusters(data)
            
            if centers is None:
                continue
            
            # Упорядочиваем центры
            if centers[0] > centers[1]:
                centers = centers[::-1]
                sigmas = sigmas[::-1]
            
            # Позиция разреза - середина между центрами
            cut_val = int(np.floor((centers[0] + centers[1]) / 2.0))
            #mid_world = 0.5 * (centers[0] + centers[1]) * self.config.scale_fixed  # TODO: use actual scale
            mid_world = 0.5 * (centers[0] + centers[1]) * scale
            
            # Метрика разделимости
            separation = self._compute_separation(centers, sigmas)
            
            # Статистический тест (если доступен)
            p_value = self._test_normality(data) if self._anderson else 1.0
            
            # Критерий принятия
            pass_test = (self._anderson and p_value < self.config.gmeans_alpha) or \
                       (not self._anderson and separation > 3.0)
            
            if pass_test and separation > best_separation:
                best_separation = separation
                best_axis = axis
                best_cut = cut_val
                best_mid_world = float(mid_world)
        
        trigger = best_axis >= 0
        return trigger, best_axis, best_cut, best_mid_world
    
    def _find_clusters(self, data: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Поиск двух кластеров в одномерных данных"""
        if self._kmeans:
            # Используем KMeans
            km = self._kmeans(n_clusters=2, n_init=2, random_state=self.random_state)
            labels = km.fit_predict(data.reshape(-1, 1))
            centers = km.cluster_centers_.flatten()
            
            sigmas = np.zeros(2)
            for k in range(2):
                cluster_data = data[labels == k]
                sigmas[k] = cluster_data.std(ddof=1) if cluster_data.size > 1 else 1e-8
        else:
            # Фолбэк на медианное разбиение
            median_val = np.median(data)
            cluster1 = data[data <= median_val]
            cluster2 = data[data > median_val]
            
            if cluster1.size == 0 or cluster2.size == 0:
                return None, None
            
            centers = np.array([cluster1.mean(), cluster2.mean()])
            sigmas = np.array([
                cluster1.std(ddof=1) if cluster1.size > 1 else 1e-8,
                cluster2.std(ddof=1) if cluster2.size > 1 else 1e-8
            ])
        
        return centers, sigmas + 1e-8
    
    def _compute_separation(self, centers: np.ndarray, sigmas: np.ndarray) -> float:
        """Вычисление метрики разделимости кластеров"""
        distance = abs(centers[1] - centers[0])
        pooled_sigma = np.sqrt((sigmas[0]**2 + sigmas[1]**2) / 2.0)
        return distance / (pooled_sigma + 1e-8)
    
    def _test_normality(self, data: np.ndarray) -> float:
        """Anderson-Darling тест на нормальность"""
        if not self._anderson:
            return 1.0
        
        # Стандартизация данных
        z = (data - data.mean()) / (data.std(ddof=1) + 1e-8)
        
        # Тест
        result = self._anderson(z)
        
        # Определение p-value
        p_value = 1.0
        for critical_val, significance in zip(result.critical_values, result.significance_level):
            if result.statistic > critical_val:
                p_value = significance / 100.0
                break
        
        return p_value
