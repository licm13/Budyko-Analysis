"""Public API for Budyko formulations."""

try:
    from .curves import (
        BudykoCurves,
        PotentialEvaporation,
        cheng_baseflow_ratio,
        fu_zhang_runoff_ratio,
        invert_cheng_qbp,
        invert_fu_zhang_alpha,
    )
    
    __all__ = [
        "BudykoCurves",
        "PotentialEvaporation",
        "cheng_baseflow_ratio",
        "fu_zhang_runoff_ratio",
        "invert_cheng_qbp",
        "invert_fu_zhang_alpha",
    ]
except ImportError:
    __all__ = []

