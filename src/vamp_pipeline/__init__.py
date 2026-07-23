"""
VAMP Engine - Core Source Package
---------------------------------
This package contains the modular engine components for the VAMP routing, 
actuarial extrapolation, and time-aware allocation pipeline.
"""

# 1. Define package metadata
__version__ = "3.0.0"
__author__ = "VAMP Engine Team"

# 2. Hoist the main classes and utilities to the package level
from .utils import setup_logger, load_config
from .data_extractor import DataExtractor
from .actuarial_engine import ActuarialEngine
from .allocation_engine import AllocationEngine
from .export_manager import ExportManager

# 3. Explicitly declare what is available when someone imports from `src`
__all__ = [
    "setup_logger",
    "load_config",
    "DataExtractor",
    "ActuarialEngine",
    "AllocationEngine",
    "ExportManager"
]