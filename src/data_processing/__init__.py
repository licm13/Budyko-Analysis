"""Top-level exports for the data_processing package.

This package contains utilities for preparing basin-level inputs and
loading auxiliary datasets (GRACE, LAI, CO2, and optionally CMIP6).
"""

try:
    from .basin_processor import BasinDataProcessor
    from .budyko_ml import (
        BudykoMLColumnMap,
        BudykoMLPreprocessor,
        BudykoMLPreprocessorConfig,
        QCResult,
    )
    from .grace_lai_processor import GRACEDataLoader, LAIDataLoader, CO2DataLoader
    
    # CMIP6 support may be optional depending on user environment
    try:
        from .cmip6_processor import CMIP6Processor  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        CMIP6Processor = None  # type: ignore[assignment]
    
    __all__ = [
        "BasinDataProcessor",
        "BudykoMLColumnMap",
        "BudykoMLPreprocessor",
        "BudykoMLPreprocessorConfig",
        "GRACEDataLoader",
        "LAIDataLoader",
        "CO2DataLoader",
        "QCResult",
    ]
    
    if CMIP6Processor is not None:  # type: ignore[name-defined]
        __all__.append("CMIP6Processor")
        
except ImportError:
    # Allow individual module imports when package structure causes issues
    __all__ = []

__all__ = [
    "BasinDataProcessor",
    "BudykoMLColumnMap",
    "BudykoMLPreprocessor",
    "BudykoMLPreprocessorConfig",
    "GRACEDataLoader",
    "LAIDataLoader",
    "CO2DataLoader",
    "QCResult",
]

if CMIP6Processor is not None:  # type: ignore[name-defined]
    __all__.append("CMIP6Processor")
