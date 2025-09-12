#!/usr/bin/env python
"""
BVH Partition - Точка входа для CLI

Адаптивное разбиение облака точек на блоки по локальной плотности
"""
import psutil
import os
import argparse
import logging
import sys
import time
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def setup_logging(log_path: Path, verbose: bool):
    """Настраивает раздельное логирование в файл и консоль."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Уровень для файла всегда DEBUG, для консоли - в зависимости от флага --verbose
    console_level = logging.DEBUG if verbose else logging.INFO
    file_level = logging.DEBUG

    # Создаем главный логгер
    logger = logging.getLogger() # Получаем корневой логгер
    logger.setLevel(logging.DEBUG) # Устанавливаем минимальный уровень обработки

    # Убираем все предыдущие обработчики, чтобы избежать дублирования
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Обработчик для файла (всегда пишет DEBUG)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Обработчик для консоли (INFO или DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# Импорты модулей проекта
from bvh_partition import __version__
from bvh_partition.config import BVHConfig
from bvh_partition.core.builder import BVHBuilder
from bvh_partition.io.loaders import load_point_cloud, validate_point_cloud
from bvh_partition.io.exporters import export_leaves, export_statistics
from bvh_partition.utils.density import estimate_scale, compute_density_stats
from bvh_partition.visualization.tracer import TraceRecorder


def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='BVH Partition - Адаптивное разбиение облака точек',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Примеры использования:
    %(prog)s --input cloud.las --output out/ --export las json
    %(prog)s --input cloud.xyz --max-points 100000 --density-grid xy
    %(prog)s --input cloud.ply --scale-mode knn --trace-json
            """
    )
    
    # Основные параметры
    parser.add_argument('--input', '-i', required=True, type=str,
                       help='Путь к входному облаку точек')
    parser.add_argument('--output', '-o', default='output', type=str,
                       help='Выходная директория (по умолчанию: output)')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    # Экспорт
    parser.add_argument('--export', nargs='+', 
                       choices=['none', 'json', 'las', 'ply', 'xyz', 'txt', 'same'],
                       default=['json'],
                       help='Форматы экспорта (можно несколько)')
    
    # Размеры узлов
    group_size = parser.add_argument_group('Размеры узлов')
    group_size.add_argument('--max-points', type=int, default=50000,
                           help='Максимум точек в листе (по умолчанию: 50000)')
    group_size.add_argument('--min-points', type=int, default=5000,
                           help='Минимум точек в листе (по умолчанию: 5000)')
    
    # Масштаб сетки
    group_scale = parser.add_argument_group('Масштаб сетки')
    group_scale.add_argument('--scale-mode', 
                            choices=['auto', 'xy', 'knn', 'fixed'],
                            default='auto',
                            help='Метод выбора шага сетки')
    group_scale.add_argument('--scale-fixed', type=float,
                            help='Фиксированный шаг сетки в метрах (для --scale-mode fixed)')
    group_scale.add_argument('--scale-kappa', type=float, default=15.0,
                            help='Коэффициент масштабирования (по умолчанию: 15.0)')
    
    # Плотностной режим
    group_density = parser.add_argument_group('Плотностной режим')
    group_density.add_argument('--density-grid',
                               choices=['off', 'xy', 'xz', 'yz'],
                               default='off',
                               help='Плоскость для анализа плотности')
    group_density.add_argument('--grid-kappa', type=int,
                               help='Целевое число точек в 2D-ячейке')
    group_density.add_argument('--target-ppm2', type=float,
                               help='Целевая плотность точек/м²')
    group_density.add_argument('--lambda-D', type=float, default=0.7,
                               help='Вес метрики плотности (по умолчанию: 0.7)')
    group_density.add_argument('--lambda-H', type=float, default=0.15,
                               help='Вес метрики высоты (по умолчанию: 0.15)')
    
    # Параметры алгоритма
    group_algo = parser.add_argument_group('Параметры алгоритма')
    group_algo.add_argument('--sah-bins', type=int, default=32,
                           help='Количество бинов для SAH (по умолчанию: 32)')
    group_algo.add_argument('--sah-epsilon', type=float, default=0.02,
                           help='Минимальное улучшение SAH (по умолчанию: 0.02)')
    group_algo.add_argument('--no-gmeans', action='store_true',
                           help='Отключить G-means детектор мультимодальности')
    group_algo.add_argument('--gmeans-alpha', type=float,
                           help='Уровень значимости для теста G-means (по умолчанию: 1.0)')
    group_algo.add_argument('--gmeans-override', type=float,
                           help='Порог SAH Gain, чтобы он был предпочтен G-means (например, 0.8)')
    group_algo.add_argument('--no-merge', action='store_true',
                           help='Отключить слияние мелких листьев')
    group_algo.add_argument('--random-seed', type=int, default=42,
                           help='Seed для воспроизводимости (по умолчанию: 42)')
    
    # Визуализация и отладка
    group_debug = parser.add_argument_group('Визуализация и отладка')
    group_debug.add_argument('--trace-json', action='store_true',
                            help='Сохранить JSON трассировку для визуализации')
    group_debug.add_argument('--trace-sample', type=int, default=8000,
                            help='Размер выборки для трассировки')
    group_debug.add_argument('--trace-grid-lines', type=int, default=8,
                            help='Количество линий сетки в трассировке')
    group_debug.add_argument('--stats', action='store_true',
                            help='Экспортировать статистику построения')
    group_debug.add_argument('--verbose', '-v', action='store_true',
                            help='Подробный вывод')
    group_debug.add_argument('--quiet', '-q', action='store_true',
                            help='Минимальный вывод')
    group_debug.add_argument('--stats-csv', action='store_true',
                            help='Сохранить CSV файл со статистикой по каждому разбиению')
    
    # Дополнительные опции
    parser.add_argument('--config', type=str,
                       help='Путь к файлу конфигурации JSON')
    parser.add_argument('--save-config', type=str,
                       help='Сохранить текущую конфигурацию в файл')
    
    return parser.parse_args()

def main():
    """Основная функция"""
    args = parse_arguments()
    output_dir = Path(args.output)
    log_file_path = output_dir / 'build_log.txt'
    setup_logging(log_file_path, args.verbose)
    logger.info(f"Detailed logs are being saved to {log_file_path}")
    process = psutil.Process(os.getpid())
    try:
        # ============ 1. Загрузка конфигурации ============
        if args.config:
            logger.info(f"Loading config from {args.config}")
            config = BVHConfig.load(Path(args.config))
            # Обновление из командной строки
            config = BVHConfig.from_args(args)
        else:
            config = BVHConfig.from_args(args)
        
        # Дополнительные настройки из аргументов
        if args.no_gmeans:
            config.gmeans_enabled = False
        if args.no_merge:
            config.merge_enabled = False
        
        # Валидация конфигурации
        config.validate()
        
        # Сохранение конфигурации если требуется
        if args.save_config:
            config.save(Path(args.save_config))
            logger.info(f"Config saved to {args.save_config}")
        
        # ============ 2. Загрузка данных ============
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Loading point cloud from {input_path.name}")
        start_time = time.perf_counter()
        
        points, metadata = load_point_cloud(input_path.as_posix())
        load_time = time.perf_counter() - start_time
        
        logger.info(
            f"Loaded {points.shape[0]:,} points in {load_time:.2f}s "
            f"({points.shape[0] / load_time / 1e6:.2f} Mpts/s)"
        )
        
        # Валидация облака точек
        validate_point_cloud(points)
        
        # ============ 3. Вычисление масштаба сетки ============
        scale_m = estimate_scale(
            points,
            config.scale_mode,
            config.scale_fixed,
            config.scale_kappa
        )
        
        # Статистика плотности
        if args.verbose:
            density_stats = compute_density_stats(points, scale_m)
            logger.debug(f"Density stats: {density_stats}")
        
        # ============ 4. Настройка трассировки и статистики ============
        trace = None
        if args.trace_json or args.stats_csv:
            trace = TraceRecorder(
                max_points_sample=args.trace_sample,
                grid_lines_target=args.trace_grid_lines
            )
            logger.info("Trace/Stats recorder enabled")
       
        if args.stats_csv:
            stats_csv_file = output_dir / 'split_statistics.csv'
            trace.start_stats_recording(stats_csv_file)

        
        # ============ 5. Построение BVH дерева ============
        logger.info("Building BVH tree...")
        builder = BVHBuilder(config, trace)
        cpu_time_before = process.cpu_times()
       
        start_time = time.perf_counter()
        tree_root = builder.build(points, scale_m)
        build_time = time.perf_counter() - start_time
        
        cpu_time_after = process.cpu_times()
        # Вычисляем использованное процессорное время
        cpu_time_sec = (cpu_time_after.user - cpu_time_before.user) + (cpu_time_after.system - cpu_time_before.system)
        
        # Замеряем пиковое использование памяти (в мегабайтах)
        # Примечание: это замеряет текущее потребление после пика, что обычно очень близко к нему.
        peak_memory_mb = process.memory_info().rss / (1024 * 1024)
        # Статистика дерева
        tree_stats = tree_root.get_stats()
        logger.info(
            f"BVH built in {build_time:.2f}s: "
            f"{tree_stats['node_count']} nodes, "
            f"{tree_stats['leaf_count']} leaves, "
            f"depth={tree_stats['depth']}"
        )
        
        if args.verbose:
            logger.debug(f"Leaf sizes: min={tree_stats['min_leaf_size']}, "
                        f"max={tree_stats['max_leaf_size']}, "
                        f"median={tree_stats['median_leaf_size']}")
        
        # ============ 6. Экспорт результатов ============
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Экспорт блоков
        export_formats = [fmt for fmt in args.export if fmt != 'none']
        if export_formats:
            logger.info(f"Exporting to formats: {', '.join(export_formats)}")
            export_leaves(
                tree_root, points, output_dir,
                export_formats, metadata, scale_m
            )
        
        # Экспорт трассировки
        if args.trace_json and trace:
            trace_file = output_dir / 'trace.json'
            trace.dump(trace_file)
            logger.info(f"Trace saved to {trace_file}")
        
        # Экспорт статистики
        if args.stats:
            stats_file = output_dir / 'statistics.json'
            export_statistics(
                tree_root, points, stats_file,
                build_time, peak_memory_mb, cpu_time_sec, config
            )
            logger.info(f"Statistics saved to {stats_file}")
        
        # ============ 7. Итоговая информация ============
        logger.info("=" * 60)
        logger.info("BVH Partition completed successfully!")
        logger.info(f"Input: {points.shape[0]:,} points from {input_path.name}")
        logger.info(f"Output: {tree_stats['leaf_count']} blocks in {output_dir}")
        logger.info(f"Total time: {load_time + build_time:.2f}s")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 2
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return 3
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 255


if __name__ == '__main__':
    sys.exit(main())
