"""
Setup script for BVH Partition package
"""
from setuptools import setup, find_packages
from pathlib import Path

# Чтение README для long_description
this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')
else:
    long_description = "BVH Partition - Adaptive point cloud partitioning"

# Чтение версии из пакета
version = "1.0.0"
version_file = this_directory / "bvh_partition" / "__init__.py"
if version_file.exists():
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                version = line.split('=')[1].strip().strip('"').strip("'")
                break

setup(
    name="bvh-partition",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="Adaptive point cloud partitioning using BVH with density-aware splitting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bvh-partition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    
    # Зависимости
    install_requires=[
        "numpy>=1.21.0",
    ],
    
    # Опциональные зависимости
    extras_require={
        "las": ["laspy[lazrs]>=2.0.0"],
        "ply": ["open3d>=0.15.0"],
        "full": [
            "laspy[lazrs]>=2.0.0",
            "open3d>=0.15.0",
            "scikit-learn>=1.0.0",
            "scipy>=1.7.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    
    # Точка входа для CLI
    entry_points={
        "console_scripts": [
            "bvh-partition=main:main",
        ],
    },
    
    # Включить package data
    include_package_data=True,
    zip_safe=False,
)
