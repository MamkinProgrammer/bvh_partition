# BVH Partition

Адаптивное разбиение облака точек на блоки с учётом локальной плотности на основе BVH (Bounding Volume Hierarchy).

## Возможности

- Поддержка форматов: LAS/LAZ, PLY, XYZ, TXT, PTS
- Адаптивное разбиение по плотности (2D/3D)
- Оптимизация по Surface Area Heuristic (SAH)
- Детекция мультимодальности (G-means)
- Слияние мелких блоков
- Экспорт в множественные форматы
- Трассировка для визуализации

## Установка

### Минимальная установка (только текстовые форматы)
```bash
pip install numpy
```

### Полная установка (все форматы)
```bash
pip install -r requirements.txt
```

# Или обычная установка
pip install .
```

## Быстрый старт

### Базовое использование
```bash
python main.py --input cloud.las --output output/
```

### С настройкой параметров
```bash
python main.py \
    --input cloud.xyz \
    --output results/ \
    --max-points 100000 \
    --min-points 10000 \
    --export las json \
    --scale-mode auto
```

### Плотностной режим (для LiDAR данных)
```bash
python main.py \
    --input lidar.laz \
    --output blocks/ \
    --density-grid xy \
    --target-ppm2 100 \
    --lambda-D 0.7 \
    --export las
```

### С трассировкой для визуализации
```bash
python main.py \
    --input data.ply \
    --output viz/ \
    --trace-json \
    --stats \
    --export json
```

## Параметры командной строки

### Основные
- `--input FILE` - Входной файл облака точек
- `--output DIR` - Выходная директория
- `--export FORMAT` - Форматы экспорта: json, las, ply, xyz, txt, same

### Размеры блоков
- `--max-points N` - Максимум точек в блоке (50000)
- `--min-points N` - Минимум точек в блоке (5000)

### Масштаб сетки
- `--scale-mode MODE` - Метод выбора шага: auto, xy, knn, fixed
- `--scale-fixed VALUE` - Фиксированный шаг в метрах
- `--scale-kappa VALUE` - Коэффициент масштабирования (15.0)

### Плотностной режим
- `--density-grid PLANE` - Плоскость анализа: off, xy, xz, yz
- `--target-ppm2 VALUE` - Целевая плотность точек/м²
- `--lambda-D VALUE` - Вес метрики плотности (0.7)
- `--lambda-H VALUE` - Вес метрики высоты (0.15)

### Алгоритм
- `--sah-bins N` - Количество бинов для SAH (32)
- `--sah-epsilon VALUE` - Порог улучшения SAH (0.02)
- `--no-gmeans` - Отключить G-means детектор
- `--no-merge` - Отключить слияние мелких блоков

### Отладка
- `--trace-json` - Сохранить трассировку
- `--stats` - Экспортировать статистику
- `--verbose` - Подробный вывод
- `--quiet` - Минимальный вывод

## Структура проекта

```
bvh_partition/
├── __init__.py              # Инициализация пакета
├── config.py                # Конфигурация и константы
├── core/                    # Ядро алгоритма
│   ├── structures.py        # Структуры данных (AABB, Node)
│   ├── builder.py           # Построитель BVH дерева
│   └── metrics.py           # Метрики разбиения
├── io/                      # Ввод-вывод
│   ├── loaders.py          # Загрузчики форматов
│   └── exporters.py        # Экспортеры результатов
├── utils/                   # Утилиты
│   └── density.py          # Оценка плотности
└── visualization/           # Визуализация
    └── tracer.py           # Трассировщик процесса
```

## Алгоритм

### 1. Квантизация пространства
Координаты точек квантуются в целочисленную сетку:
```
G(p) = floor(p / scale_m)
```

### 2. Рекурсивное разбиение
Критерий остановки:
```
n_node <= rho_target * V_node^gamma
```

### 3. Метрики разбиения

**Surface Area Heuristic (SAH):**
```
Cost_SAH = (SA_left * n_left + SA_right * n_right) / SA_parent
```

**Density Gap:**
```
DensityGap = |ρ_left - ρ_right| / T_ppm2
```

**Height Contrast:**
```
HeightContrast = |μ_z_left - μ_z_right| / (z_max - z_min)
```

**Итоговая метрика:**
```
Cost = Cost_SAH - λ_D * DensityGap - λ_H * HeightContrast
```

## Примеры конфигурации

### Для городского LiDAR
```bash
python main.py \
    --input city.laz \
    --density-grid xy \
    --target-ppm2 50 \
    --max-points 200000 \
    --scale-mode xy
```

### Для лесного LiDAR
```bash
python main.py \
    --input forest.las \
    --density-grid xy \
    --lambda-H 0.3 \
    --max-points 100000
```

### Для фотограмметрии
```bash
python main.py \
    --input photogrammetry.ply \
    --scale-mode knn \
    --scale-kappa 20 \
    --no-gmeans
```

## API использование

```python
from bvh_partition import BVHConfig, BVHBuilder
from bvh_partition.io import load_point_cloud
from bvh_partition.utils import estimate_scale

# Загрузка данных
points, metadata = load_point_cloud("cloud.las")

# Конфигурация
config = BVHConfig(
    max_points_per_leaf=100000,
    min_points_per_leaf=10000,
    density_mode='xy',
    target_ppm2=100
)

# Вычисление масштаба
scale_m = estimate_scale(points, mode='auto')

# Построение дерева
builder = BVHBuilder(config)
tree = builder.build(points, scale_m)

# Статистика
stats = tree.get_stats()
print(f"Leaves: {stats['leaf_count']}, Depth: {stats['depth']}")
```

## Производительность

Типичная скорость обработки:
- Загрузка: 5-10 M точек/сек
- Построение: 1-3 M точек/сек
- Экспорт: 2-5 M точек/сек

Рекомендации:
- Используйте conda для numpy/scipy (ускорение до 2x)
- Для больших облаков (>100M) используйте подвыборку
- Оптимальный `--sah-bins` = 16-64

## Поддерживаемые форматы

### Входные
| Формат | Расширение | Требования | Атрибуты |
|--------|-----------|------------|----------|
| LAS/LAZ | .las, .laz | laspy | RGB, intensity, classification |
| PLY | .ply | open3d | RGB, normals |
| XYZ/TXT | .xyz, .txt, .pts | - | Дополнительные столбцы |

### Выходные
| Формат | Описание | Сохраняет атрибуты |
|--------|----------|-------------------|
| JSON | Метаданные блоков | Нет |
| LAS | Исходный формат | Да |
| PLY | С цветами | Частично |
| XYZ | Текстовый | Да |

