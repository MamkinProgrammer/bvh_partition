"""
Конфигурация и константы для BVH разбиения
"""
from dataclasses import dataclass, field
from typing import Optional, Literal
import json
from pathlib import Path

# ============ КОНСТАНТЫ ============

# Размеры узлов
MIN_CHILD_FRACTION = 0.35 # Минимальная доля точек в дочернем узле от динамического максимума
DEFAULT_GRID_CELLS = 20000  # Количество ячеек для автоматического масштаба сетки

# Плотность
DENSITY_TARGET_FACTOR = 0.9  # Коэффициент для вычисления целевой плотности от измеренной
DENSITY_CELL_SIZE = 0.5  # Размер ячейки (м) для оценки плотности

# Эпсилоны и пороги
GRID_EPSILON = 1e-6  # Эпсилон для операций с сеткой
SCALE_EPSILON = 1e-3  # Минимальный масштаб сетки (м)
VOLUME_EPSILON = 1e-9  # Минимальный объём для вычислений

# Сэмплирование
KNN_K = 8  # Количество соседей для kNN spacing
KNN_SAMPLE_SIZE = 200_000  # Размер выборки для kNN
GMEANS_SAMPLE_SIZE = 200_000  # Размер выборки для G-means теста
TRACE_SAMPLE_SIZE = 8000  # Размер выборки для визуализации

# Визуализация
TRACE_GRID_LINES = 8  # Количество линий сетки для визуализации


@dataclass
class BVHConfig:
    """Конфигурация построителя BVH"""
    
    # ======== Ограничения на размер узла ========
    max_points_per_leaf: int = 50_000
    min_points_per_leaf: int = 5_000
    
    # ======== Параметры адаптивной плотности ========
    rho_target: float = 4e-5  # Базовая плотность для динамического порога
    gamma: float = 0.68  # Показатель степени для масштабирования порога
    
    # ======== SAH (Surface Area Heuristic) ========
    sah_bins: int = 32  # Количество бинов для поиска разреза
    sah_epsilon: float = 0.02  # Минимальное улучшение для применения SAH
    
    # ======== G-means кластеризация ========
    gmeans_alpha: Optional[float] = None # Уровень значимости для Anderson-Darling теста
    gmeans_enabled: bool = True  # Использовать G-means для обнаружения мультимодальности
    gmeans_override: Optional[float] = None # Порог Gain для SAH, чтобы он был предпочтен G-means
    
    # ======== Слияние мелких узлов ========
    merge_enabled: bool = True  # Включить слияние мелких листьев
    merge_beta: float = 0.4  # Не используется в текущей версии
    
    # ======== Режимы работы с плотностью ========
    grid_mode: Literal['off', 'xy', 'xz', 'yz'] = 'off'  # Плоскость для сетки
    density_mode: Literal['off', 'xy', 'xz', 'yz', '3d'] = 'off'  # Плоскость для плотности
    
    # ======== Параметры плотностной сетки ========
    grid_kappa: Optional[int] = None  # Целевое число точек в 2D-ячейке
    target_ppm2: Optional[float] = None  # Целевая плотность точек/м²
    target_ppv3: Optional[float] = None  # Целевая плотность точек/м³ (для 3D режима)
    
    # ======== Веса для комбинированной метрики ========
    lambda_density: float = 0.7  # Вес DensityGap в итоговой метрике
    lambda_height: float = 0.15  # Вес HeightContrast в итоговой метрике
    
    # ======== Масштаб сетки ========
    scale_mode: Literal['auto', 'xy', 'knn', 'fixed'] = 'auto'
    scale_fixed: Optional[float] = None  # Фиксированный шаг сетки (м)
    scale_kappa: float = 15.0  # Делитель для вычисления шага из spacing
    
    # ======== Прочие параметры ========
    random_seed: int = 42  # Seed для воспроизводимости
    min_height: int = 2  # Минимальная высота узла в ячейках сетки
    log_density: bool = True  # Логировать информацию о плотности
    
    # ======== Трассировка для визуализации ========
    trace_enabled: bool = False
    trace_sample_size: int = TRACE_SAMPLE_SIZE
    trace_grid_lines: int = TRACE_GRID_LINES
    
    def validate(self) -> None:
        """Проверка корректности конфигурации"""
        if self.max_points_per_leaf < self.min_points_per_leaf:
            raise ValueError(
                f"max_points_per_leaf ({self.max_points_per_leaf}) "
                f"должен быть >= min_points_per_leaf ({self.min_points_per_leaf})"
            )
        
        if self.scale_mode == 'fixed' and (self.scale_fixed is None or self.scale_fixed <= 0):
            raise ValueError("Для scale_mode='fixed' требуется положительное значение scale_fixed")
        
        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError(f"gamma должна быть в диапазоне (0, 1], получено: {self.gamma}")
        
        if self.sah_bins < 2:
            raise ValueError(f"sah_bins должно быть >= 2, получено: {self.sah_bins}")
    
    def save(self, path: Path) -> None:
        """Сохранение конфигурации в JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> 'BVHConfig':
        """Загрузка конфигурации из JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_args(cls, args) -> 'BVHConfig':
        """Создание конфигурации из аргументов командной строки"""
        config = cls()
        
        # Обновляем из аргументов
        if hasattr(args, 'max_points'):
            config.max_points_per_leaf = args.max_points
        if hasattr(args, 'min_points'):
            config.min_points_per_leaf = args.min_points
        
        if hasattr(args, 'gmeans_alpha') and args.gmeans_alpha is not None:
            config.gmeans_alpha = args.gmeans_alpha
            
        if hasattr(args, 'gmeans_override') and args.gmeans_override is not None:
            config.gmeans_override = args.gmeans_override
        
        if hasattr(args, 'scale_mode'):
            config.scale_mode = args.scale_mode
        if hasattr(args, 'scale_fixed'):
            config.scale_fixed = args.scale_fixed
        if hasattr(args, 'kappa_scale'):
            config.scale_kappa = args.kappa_scale
            
        if hasattr(args, 'density_grid'):
            config.density_mode = args.density_grid
            config.grid_mode = args.density_grid
        if hasattr(args, 'grid_kappa'):
            config.grid_kappa = args.grid_kappa
        if hasattr(args, 'target_ppm2'):
            config.target_ppm2 = args.target_ppm2
        if hasattr(args, 'lambda_D'):
            config.lambda_density = args.lambda_D
        if hasattr(args, 'lambda_H'):
            config.lambda_height = args.lambda_H
            
        if hasattr(args, 'trace_json'):
            config.trace_enabled = args.trace_json
        if hasattr(args, 'trace_sample'):
            config.trace_sample_size = args.trace_sample
        if hasattr(args, 'trace_grid_lines'):
            config.trace_grid_lines = args.trace_grid_lines
        
        config.validate()
        return config
