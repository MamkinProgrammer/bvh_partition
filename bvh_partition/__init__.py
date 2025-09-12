from __future__ import annotations
from typing import Any

# 1) Версия пакета
# В сборке из PyPI можно читать метаданные:
try:
    from importlib.metadata import version as _pkg_version  # Py>=3.8
except Exception:  # на всякий случай для старых окружений
    _pkg_version = None  # type: ignore

try:
    # имя дистрибутива — как в pyproject/setup (если будет)
    __version__ = _pkg_version("bvh-partition") if _pkg_version else "0.1.0"
except Exception:
    # в editable/develop-режиме пакет может быть не «установлен»
    __version__ = "0.1.0"

__all__ = ["__version__", "BVHBuilder"]

# 2) Ленивый экспорт для публичного API (избегаем ранних импортов)
def __getattr__(name: str) -> Any:
    if name == "BVHBuilder":
        from .core.builder import BVHBuilder  # локальный импорт, не создаёт цикл
        return BVHBuilder
    raise AttributeError(name)
