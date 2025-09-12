import numpy as np
import time
from typing import Tuple, Optional
import logging

from .structures import AABB, Node
from .metrics import MetricsCalculator, GmeansEvaluator, SplitCandidate
from ..config import BVHConfig, MIN_CHILD_FRACTION, GMEANS_SAMPLE_SIZE
from ..visualization.tracer import TraceRecorder

logger = logging.getLogger(__name__)


class BVHBuilder:
    """Построитель BVH дерева с адаптивным разбиением по плотности"""
    
    def __init__(self, config: BVHConfig, trace: Optional[TraceRecorder] = None):
        """
        Args:
            config: Конфигурация 
            trace: Опциональный трассировщик для визуализации
        """
        self.config = config
        self.trace = trace
        
        # Инициализация компонентов
        self.metrics = MetricsCalculator(config)
        self.gmeans = GmeansEvaluator(config)
        self.rng = np.random.default_rng(config.random_seed)
        
        # Масштаб сетки (устанавливается в build)
        self.scale_m = 1.0
        
        # Статистика построения
        self.stats = {
            'build_time': 0.0,
            'nodes_created': 0,
            'splits_performed': 0,
            'merges_performed': 0
        }
    
    def build(self, points: np.ndarray, scale_m: float) -> Node:
        """
        Построение BVH дерева
        
        Args:
            points: Облако точек (N x 3) в метрах
            scale_m: Масштаб сетки (метры на ячейку)
        
        Returns:
            Корневой узел построенного дерева
        """
        start_time = time.perf_counter()
        
        self.scale_m = scale_m
        self.metrics.scale_m = scale_m  # Важно: обновляем масштаб в метриках
        
        # Квантизация в целочисленную сетку
        grid_coords = np.floor(points / scale_m).astype(np.int64)
        
        # Определение границ
        min_corner = grid_coords.min(axis=0)
        max_corner = grid_coords.max(axis=0) + 1
        
        # ИСПРАВЛЕНИЕ: Автоматически включаем метрики если заданы веса
        if self.config.lambda_density > 0 and self.config.density_mode == 'off':
            self.config.density_mode = 'xy'  # По умолчанию используем XY проекцию
            logger.info(f"Auto-enabled density_mode='xy' due to lambda_density={self.config.lambda_density}")
        
        # Трассировка сетки
        if self.trace:
            self.trace.record_input_points(points[:, :2])
            self.trace.record_grid_lines(min_corner, max_corner, scale_m)
        
        # Создание корневого узла
        root_aabb = AABB(
            min_corner, max_corner,
            closed=(True, False, True, False, True, False)
        )
        root_indices = np.arange(points.shape[0], dtype=np.int64)
        root_node = Node(root_aabb, root_indices, level=0)
        
        logger.info(f"Building BVH for {points.shape[0]} points with scale={scale_m:.6f}m")
        logger.info(f"Density mode: {self.config.density_mode}, λ_D={self.config.lambda_density}, λ_H={self.config.lambda_height}")
        
        # Рекурсивное построение
        self.stats['nodes_created'] = 0
        self.stats['splits_performed'] = 0
        root_node = self._split_node(root_node, grid_coords, points)
        
        # Слияние мелких листьев
        if self.config.merge_enabled:
            self.stats['merges_performed'] = 0
            self._merge_small_leaves(root_node)
            logger.info(f"Performed {self.stats['merges_performed']} leaf merges")
        
        # Финальная трассировка
        if self.trace:
            self.trace.record_final_boxes(root_node.iter_leaves(), scale_m)
            self.trace.close()
        
        self.stats['build_time'] = time.perf_counter() - start_time
        
        # Логирование статистики
        tree_stats = root_node.get_stats()
        logger.info(
            f"BVH built in {self.stats['build_time']:.2f}s: "
            f"{tree_stats['node_count']} nodes, "
            f"{tree_stats['leaf_count']} leaves, "
            f"depth={tree_stats['depth']}"
        )
        
        return root_node
    
    def _split_node(self,
                    node: Node,
                    grid_coords: np.ndarray,
                    world_coords: np.ndarray) -> Node:
        """
        Рекурсивное разбиение узла
        
        Args:
            node: Текущий узел
            grid_coords: Координаты всех точек в сетке
            world_coords: Координаты всех точек в метрах
        
        Returns:
            Обновлённый узел (с детьми или как лист)
        """
        self.stats['nodes_created'] += 1
        
        # Проверка необходимости разбиения
        n_pts = node.point_count()
        volume = node.aabb.volume()
        dimensions = node.aabb.dimensions()
        
        if not self.metrics.should_split(n_pts, volume, dimensions):
            return node
        
        # Трассировка выбранного узла
        if self.trace and node.level == 0:
            self.trace.record_density_box(node.aabb, self.scale_m)
        
        # Поиск оптимального разбиения
        split_candidate = self._find_best_split(node, grid_coords, world_coords)
        
        if split_candidate is None:
            return node
        
        # Выполнение разбиения
        left_node, right_node = self._perform_split(
            node, split_candidate, grid_coords, world_coords
        )
        
        if left_node is None or right_node is None:
            return node
        
        # Трассировка разбиения
        if self.trace and node.level == 0:
            self.trace.record_split(
                node.aabb,
                split_candidate.axis,
                split_candidate.position,
                self.scale_m
            )
        
        self.stats['splits_performed'] += 1
        
        # Рекурсивное разбиение детей
        left_node = self._split_node(left_node, grid_coords, world_coords)
        right_node = self._split_node(right_node, grid_coords, world_coords)
        
        node.children = (left_node, right_node)
        return node
    
    def _find_best_split(self,
                        node: Node,
                        grid_coords: np.ndarray,
                        world_coords: np.ndarray) -> Optional[SplitCandidate]:
        """
        Поиск оптимального разбиения узла
        Сначала проверяет G-means триггер, затем SAH с дополнительными метриками
        """
        # G-means для обнаружения мультимодальности
        gmeans_trigger, gmeans_axis, gmeans_cut, _ = self.gmeans.evaluate(
            grid_coords, node.indices, GMEANS_SAMPLE_SIZE
        )
        
        # SAH поиск
        sah_candidate = self._evaluate_sah(node, grid_coords, world_coords)
        sah_details = {}
        if sah_candidate:
            sah_details = self._recompute_costs_for_stats(node, sah_candidate, grid_coords, world_coords)
       
        # 2. G-means проверка и создание кандидата
        gmeans_candidate = None
        if self.config.gmeans_enabled:
            gmeans_trigger, gmeans_axis, gmeans_cut, _ = self.gmeans.evaluate(
                grid_coords, node.indices, GMEANS_SAMPLE_SIZE
            )
            if gmeans_trigger:
                pts_axis = grid_coords[node.indices, gmeans_axis]
                n_left = np.count_nonzero(pts_axis <= gmeans_cut)
                n_right = node.point_count() - n_left
                
                dynamic_max = self.metrics._compute_dynamic_threshold(node.aabb.volume())
                min_child_size = max(self.config.min_points_per_leaf, int(MIN_CHILD_FRACTION * dynamic_max))

                if n_left >= min_child_size and n_right >= min_child_size:
                    gmeans_candidate = SplitCandidate(
                        axis=gmeans_axis, position=gmeans_cut, cost=0.0,
                        left_count=n_left, right_count=n_right, method='gmeans'
                    )
            
        # 3. Принятие финального решения
        final_candidate = None
        decision_method = 'none (no split)'
        
        # Получаем порог из конфига, если он не задан, используем значение по умолчанию (например, 0.5)
        override_threshold = self.config.gmeans_override if self.config.gmeans_override is not None else 0.5

        if gmeans_candidate and sah_candidate:
            base_cost = float(node.point_count())
            rel_gain = self.metrics.compute_relative_gain(
                sah_base=base_cost,
                sah_candidate=sah_candidate.cost
            )
            # <<< ВОТ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: ИСПОЛЬЗУЕМ НОВЫЙ ПАРАМЕТР >>>
            if rel_gain > override_threshold:
                final_candidate = sah_candidate
                decision_method = 'sah_preferred'
                logger.debug(f"[Decision] SAH preferred over G-means (gain {rel_gain:.2f} > {override_threshold})")
            else:
                final_candidate = gmeans_candidate
                decision_method = 'gmeans_preferred'
                logger.debug(f"[Decision] G-means preferred over SAH (gain {rel_gain:.2f} <= {override_threshold})")
        elif sah_candidate:
            final_candidate = sah_candidate
            decision_method = 'sah'
        elif gmeans_candidate:
            final_candidate = gmeans_candidate
            decision_method = 'gmeans'
        
        if self.trace and final_candidate:
        # Собираем детали для статистики только если есть что записывать
            sah_details = {}
            if sah_candidate:
                sah_details = self._recompute_costs_for_stats(node, sah_candidate, grid_coords, world_coords)
        
            base_cost = float(node.point_count()) 
        
            stats_data = {
                'level': node.level,
                'parent_points': node.point_count(),
                'parent_volume_m3': node.aabb.volume() * (self.scale_m ** 3),
                'decision_method': decision_method,
                'split_axis': final_candidate.axis,
                'chosen_cost_total': final_candidate.cost,
                'cost_base_sah': base_cost,
                'sah_candidate_cost': sah_details.get('sah_cost', 0.0),
                'sah_candidate_density_gap': sah_details.get('density_gap', 0.0),
                'sah_candidate_height_contrast': sah_details.get('height_contrast', 0.0),
                'lambda_d': self.config.lambda_density,
                'lambda_h': self.config.lambda_height,
                'left_points': final_candidate.left_count,
                'right_points': final_candidate.right_count
            }
            self.trace.record_split_decision(stats_data)
        
        return final_candidate
        
        # return sah_candidate or (SplitCandidate(
        #     axis=gmeans_axis,
        #     position=gmeans_cut,
        #     cost=0.0,
        #     left_count=-1,
        #     right_count=-1,
        #     method='gmeans'
        # ) if gmeans_trigger else None)
            
    def _recompute_costs_for_stats(self, node, candidate, grid_coords, world_coords) -> dict:
        """Пересчитывает компоненты стоимости для лучшего кандидата для записи в статистику."""
        dims_m = node.aabb.dimensions().astype(np.float64) * self.scale_m
        target_density = self._compute_target_density(node, node.point_count(), dims_m)
        
        # Это частичный дубляж кода из _compute_split_cost, но он нужен для детализации
        # Размеры дочерних узлов
        left_dim = node.aabb.dimensions().copy(); left_dim[candidate.axis] = (candidate.position + 1) - node.aabb.min_corner[candidate.axis]
        right_dim = node.aabb.dimensions().copy(); right_dim[candidate.axis] = node.aabb.max_corner[candidate.axis] - (candidate.position + 1)
        
        # Площади поверхностей
        sa_left = 2 * (left_dim[0]*left_dim[1] + left_dim[0]*left_dim[2] + left_dim[1]*left_dim[2])
        sa_right = 2 * (right_dim[0]*right_dim[1] + right_dim[0]*right_dim[2] + right_dim[1]*right_dim[2])
        
        sah_cost = self.metrics.compute_sah_cost(candidate.left_count, candidate.right_count, sa_left, sa_right, node.aabb.surface_area())
        scaled_sah = sah_cost * self.scale_m**2
        
        area_left, area_right = self._compute_projection_areas(node, candidate.axis, candidate.position, dims_m)
        density_gap = self.metrics.compute_density_gap(candidate.left_count, candidate.right_count, area_left, area_right, target_density)
        
        pts_axis = grid_coords[node.indices, candidate.axis]
        mask_left = pts_axis <= candidate.position
        z_left = world_coords[node.indices[mask_left], 2]
        z_right = world_coords[node.indices[~mask_left], 2]
        z_range = np.ptp(world_coords[node.indices, 2])
        height_contrast = self.metrics.compute_height_contrast(z_left, z_right, z_range)
        
        return {
            "sah_cost": scaled_sah,
            "density_gap": density_gap,
            "height_contrast": height_contrast,
        }
    
    def _evaluate_sah(self,
                      node: Node,
                      grid_coords: np.ndarray,
                      world_coords: np.ndarray) -> Optional[SplitCandidate]:
        """
        Поиск оптимального разбиения по SAH с дополнительными метриками
        """
        best_candidate = None
        best_cost = float('inf')
        
        indices = node.indices
        n_pts = indices.size
        base_cost_for_comparison = float(n_pts)
        base_cost = node.aabb.surface_area()
                
        if base_cost <= 0:
            return None
                
        # Подготовка для дополнительных метрик
        dims_m = node.aabb.dimensions().astype(np.float64) * self.scale_m
        
        # Вычисление целевой плотности если нужно
        target_density = self._compute_target_density(node, n_pts, dims_m)
        
        # Минимальный размер дочернего узла
        volume = node.aabb.volume()
        dynamic_max = self.metrics._compute_dynamic_threshold(volume)
        min_child = max(
            self.config.min_points_per_leaf,
            int(MIN_CHILD_FRACTION * dynamic_max)
        )
        
        # Перебор осей
        for axis in range(3):
            pts_axis = grid_coords[indices, axis]
            axis_min = node.aabb.min_corner[axis]
            axis_max = node.aabb.max_corner[axis]
            
            if axis_max - axis_min <= 1:
                continue
            
            # Создание бинов для SAH
            candidates = self._generate_split_candidates(
                pts_axis, axis_min, axis_max, n_pts, min_child
            )
            
            # Оценка каждого кандидата
            for cut_pos, n_left in candidates:
                n_right = n_pts - n_left
                
                if n_left < min_child or n_right < min_child:
                    continue
                
                # Вычисление SAH
                cost = self._compute_split_cost(
                    node, axis, cut_pos, n_left, n_right,
                    grid_coords, world_coords, indices,
                    target_density, dims_m
                )
                
                # Логирование SAH
                if logger.isEnabledFor(logging.DEBUG):
                    rel_gain = self.metrics.compute_relative_gain(base_cost_for_comparison, cost)
                    logger.debug(
                        f"[SAH] Axis={axis}, Cut={cut_pos}, "
                        f"nL={n_left}, nR={n_right}, "
                        f"CostTotal={cost:.2f}, BaseCost={base_cost_for_comparison:.2f}, Gain={rel_gain:.4f}"
                    )
                
                if cost < best_cost:
                    best_cost = cost
                    best_candidate = SplitCandidate(
                        axis=axis,
                        position=cut_pos,
                        cost=cost,
                        left_count=n_left,
                        right_count=n_right,
                        method='sah'
                    )
        
        # Проверка минимального улучшения
        if best_candidate:
            rel_gain = self.metrics.compute_relative_gain(base_cost_for_comparison, best_cost)
            
            if rel_gain > self.config.sah_epsilon:
                logger.debug(f"[Split] Best split: axis={best_candidate.axis}, pos={best_candidate.position}, gain={rel_gain:.4f}")
                return best_candidate
        
        return None
    
    def _generate_split_candidates(self,
                                   pts_axis: np.ndarray,
                                   axis_min: int,
                                   axis_max: int,
                                   n_pts: int,
                                   min_child: int) -> list:
        """Генерация кандидатов на разбиение с учетом распределения точек"""
        # Гистограмма для быстрого подсчёта
        bin_edges = np.linspace(axis_min, axis_max, num=self.config.sah_bins + 1).astype(np.int64)
        counts, _ = np.histogram(pts_axis, bins=bin_edges)
        
        # Обрезаем пустые концы
        nonzero = np.nonzero(counts)[0]
        if nonzero.size < 2:
            return []
        
        start, end = nonzero[0], nonzero[-1] + 1
        counts = counts[start:end]
        bin_edges = bin_edges[start:end+1]
        
        # Префиксные суммы для быстрого подсчёта
        prefix = np.cumsum(counts)
        
        candidates = []
        for i in range(1, counts.size):
            n_left = int(prefix[i - 1])
            if min_child <= n_left <= n_pts - min_child:
                cut_val = int(bin_edges[i] - 1)
                candidates.append((cut_val, n_left))
        
        return candidates
    
    def _compute_split_cost(self,
                            node: Node,
                            axis: int,
                            cut_pos: int,
                            n_left: int,
                            n_right: int,
                            grid_coords: np.ndarray,
                            world_coords: np.ndarray,
                            indices: np.ndarray,
                            target_density: Optional[float],
                            dims_m: np.ndarray) -> float:
        """Вычисление полной стоимости разбиения с учётом всех метрик"""
        # Размеры дочерних узлов
        left_dim = node.aabb.dimensions().copy()
        left_dim[axis] = (cut_pos + 1) - node.aabb.min_corner[axis]
        
        right_dim = node.aabb.dimensions().copy()
        right_dim[axis] = node.aabb.max_corner[axis] - (cut_pos + 1)
        
        # Площади поверхностей
        sa_left = 2 * (left_dim[0]*left_dim[1] + left_dim[0]*left_dim[2] + left_dim[1]*left_dim[2])
        sa_right = 2 * (right_dim[0]*right_dim[1] + right_dim[0]*right_dim[2] + right_dim[1]*right_dim[2])
        
        # Базовый SAH
        sah_cost = self.metrics.compute_sah_cost(
            n_left, n_right, sa_left, sa_right, node.aabb.surface_area()
        )
        
        # Дополнительные метрики (если включены)
        density_gap = 0.0
        height_contrast = 0.0
        
        if target_density and self.config.density_mode != 'off' and self.config.lambda_density > 0:
            # Вычисление площадей проекций
            area_left, area_right = self._compute_projection_areas(
                node, axis, cut_pos, dims_m
            )
            
            if area_left and area_right:
                density_gap = self.metrics.compute_density_gap(
                    n_left, n_right, area_left, area_right, target_density
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  DensityGap: {density_gap:.4f} (areas: {area_left:.2f}, {area_right:.2f})")
        
        if self.config.lambda_height > 0:
            # Разделение по высоте
            pts_axis = grid_coords[indices, axis]
            mask_left = pts_axis <= cut_pos
            
            z_left = world_coords[indices[mask_left], 2]
            z_right = world_coords[indices[~mask_left], 2]
            
            if z_left.size > 0 and z_right.size > 0:
                z_range = np.ptp(world_coords[indices, 2])
                if z_range > 0:
                    height_contrast = self.metrics.compute_height_contrast(
                        z_left, z_right, z_range
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"  HeightContrast: {height_contrast:.4f}")
        
        return self.metrics.compute_combined_cost(sah_cost, density_gap, height_contrast)
    
    def _compute_target_density(self,
                                node: Node,
                                n_pts: int,
                                dims_m: np.ndarray) -> Optional[float]:
        """Вычисление целевой плотности для узла"""
        if self.config.target_ppm2:
            return self.config.target_ppm2
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем реальную плотность узла как целевую
        # Это позволит найти разрезы, которые создают контраст плотностей
        if self.config.density_mode == 'xy':
            area = dims_m[0] * dims_m[1]
            if area > 0:
                # Целевая плотность = текущая плотность узла
                return n_pts / area
        elif self.config.density_mode == 'xz':
            area = dims_m[0] * dims_m[2]
            if area > 0:
                return n_pts / area
        elif self.config.density_mode == 'yz':
            area = dims_m[1] * dims_m[2]
            if area > 0:
                return n_pts / area
        elif self.config.density_mode == '3d':
            volume = dims_m[0] * dims_m[1] * dims_m[2]
            if volume > 0:
                return n_pts / volume
        
        return None
    
    def _compute_projection_areas(self,
                                   node: Node,
                                   axis: int,
                                   cut_pos: int,
                                   dims_m: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Вычисление площадей проекций для дочерних узлов"""
                
        if self.config.density_mode == 'xy':
            if axis == 0:  # Разрез по X
                left_width = (cut_pos + 1 - node.aabb.min_corner[0]) * self.scale_m
                right_width = (node.aabb.max_corner[0] - cut_pos - 1) * self.scale_m
                return left_width * dims_m[1], right_width * dims_m[1]
            elif axis == 1:  # Разрез по Y
                left_height = (cut_pos + 1 - node.aabb.min_corner[1]) * self.scale_m
                right_height = (node.aabb.max_corner[1] - cut_pos - 1) * self.scale_m
                return dims_m[0] * left_height, dims_m[0] * right_height
            else:  # Разрез по Z - площадь не меняется
                area = dims_m[0] * dims_m[1]
                return area, area
                
        elif self.config.density_mode == 'xz':
            if axis == 0:  # Разрез по X
                left_width = (cut_pos + 1 - node.aabb.min_corner[0]) * self.scale_m
                right_width = (node.aabb.max_corner[0] - cut_pos - 1) * self.scale_m
                return left_width * dims_m[2], right_width * dims_m[2]
            elif axis == 2:  # Разрез по Z
                left_depth = (cut_pos + 1 - node.aabb.min_corner[2]) * self.scale_m
                right_depth = (node.aabb.max_corner[2] - cut_pos - 1) * self.scale_m
                return dims_m[0] * left_depth, dims_m[0] * right_depth
            else:  # Разрез по Y - площадь не меняется
                area = dims_m[0] * dims_m[2]
                return area, area
                
        elif self.config.density_mode == 'yz':
            if axis == 1:  # Разрез по Y
                left_height = (cut_pos + 1 - node.aabb.min_corner[1]) * self.scale_m
                right_height = (node.aabb.max_corner[1] - cut_pos - 1) * self.scale_m
                return left_height * dims_m[2], right_height * dims_m[2]
            elif axis == 2:  # Разрез по Z
                left_depth = (cut_pos + 1 - node.aabb.min_corner[2]) * self.scale_m
                right_depth = (node.aabb.max_corner[2] - cut_pos - 1) * self.scale_m
                return dims_m[1] * left_depth, dims_m[1] * right_depth
            else:  # Разрез по X - площадь не меняется
                area = dims_m[1] * dims_m[2]
                return area, area
                
        elif self.config.density_mode == '3d':
            # Для 3D плотности используем объёмы
            left_dim = node.aabb.dimensions().copy().astype(float)
            left_dim[axis] = (cut_pos + 1 - node.aabb.min_corner[axis])
            
            right_dim = node.aabb.dimensions().copy().astype(float)
            right_dim[axis] = (node.aabb.max_corner[axis] - cut_pos - 1)
            
            left_volume = (left_dim * self.scale_m).prod()
            right_volume = (right_dim * self.scale_m).prod()
            
            return left_volume, right_volume
        
        return None, None
    
    def _perform_split(self,
                       parent: Node,
                       candidate: SplitCandidate,
                       grid_coords: np.ndarray,
                       world_coords: np.ndarray) -> Tuple[Optional[Node], Optional[Node]]:
        """Выполнение разбиения узла"""
        indices = parent.indices
        pts_axis = grid_coords[indices, candidate.axis]
        
        # Разделение индексов
        left_mask = pts_axis <= candidate.position
        right_mask = ~left_mask
        
        if not left_mask.any() or not right_mask.any():
            return None, None
        
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]
        
        # Создание AABB для дочерних узлов
        left_aabb, right_aabb = self._create_child_aabbs(
            parent.aabb, candidate.axis, candidate.position
        )
        
        # Создание дочерних узлов
        left_node = Node(left_aabb, left_indices, parent.level + 1)
        right_node = Node(right_aabb, right_indices, parent.level + 1)
        
        return left_node, right_node
    
    def _create_child_aabbs(self,
                            parent_aabb: AABB,
                            axis: int,
                            cut_pos: int) -> Tuple[AABB, AABB]:
        """Создание AABB для дочерних узлов"""
        # Копируем флаги closed
        closed = list(parent_aabb.closed)
        left_closed = closed.copy()
        right_closed = closed.copy()
        
        # Обновляем флаги для разреза
        max_flag_idx = 2 * axis + 1
        min_flag_idx = 2 * axis
        left_closed[max_flag_idx] = True
        right_closed[min_flag_idx] = False
        
        # Создаём границы
        left_max = parent_aabb.max_corner.copy()
        left_max[axis] = cut_pos + 1
        
        right_min = parent_aabb.min_corner.copy()
        right_min[axis] = cut_pos + 1
        
        left_aabb = AABB(parent_aabb.min_corner.copy(), left_max, tuple(left_closed))
        right_aabb = AABB(right_min, parent_aabb.max_corner.copy(), tuple(right_closed))
        
        return left_aabb, right_aabb
    
    def _merge_small_leaves(self, node: Node) -> None:
        """Рекурсивное слияние мелких листовых узлов"""
        if node.is_leaf():
            return
        
        left, right = node.children
        
        # Рекурсивно обрабатываем детей
        self._merge_small_leaves(left)
        self._merge_small_leaves(right)
        
        # Проверяем возможность слияния
        if left.is_leaf() and right.is_leaf():
            if self.metrics.can_merge_leaves(
                left.point_count(),
                right.point_count(),
                node.aabb.volume()
            ):
                # Трассировка слияния
                if self.trace:
                    self.trace.record_merge(
                        [left.aabb, right.aabb],
                        node.aabb,
                        self.scale_m
                    )
                
                # Выполняем слияние
                node.indices = np.concatenate([left.indices, right.indices])
                node.children = None
                self.stats['merges_performed'] += 1