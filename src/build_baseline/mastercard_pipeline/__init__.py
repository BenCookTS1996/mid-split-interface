"""
Mastercard Engine - Core Source Package
---------------------------------------
This package contains the modular engine components for the Mastercard routing,
actuarial chargeback (CB) extrapolation, and time-aware allocation pipeline.

It mirrors src/vamp_pipeline/ (the Visa/VAMP pipeline) phase-for-phase
(DataExtractor -> ActuarialEngine -> AllocationEngine -> ExportManager), but
implements the Mastercard-specific business logic:

  * risk metric is the Mastercard CHARGEBACK count (cb_*) rather than the Visa VAMP;
  * the Visa "kill switch" zeroes BIN 4xx (Visa) forecast volume;
  * the "Mastercard Shift" offsets FP&A sales forward one month and injects the
    last completed month's real transactions as an unaltered Month 0 baseline,
    which is excluded from re-routing and appended back after allocation.
"""

# 1. Define package metadata
__version__ = "1.0.0"
__author__ = "Mastercard Engine Team"

# 2. Hoist the main classes and utilities to the package level
from .utils import setup_logger, load_config
from .data_extractor import DataExtractor
from .actuarial_engine import ActuarialEngine
from .allocation_engine import AllocationEngine
from .export_manager import ExportManager

# 3. Explicitly declare what is available when someone imports from the package
__all__ = [
    "setup_logger",
    "load_config",
    "DataExtractor",
    "ActuarialEngine",
    "AllocationEngine",
    "ExportManager",
]
