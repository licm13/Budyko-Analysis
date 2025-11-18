# CLAUDE.md - AI Assistant Guide for Budyko-Analysis

**Last Updated:** 2025-11-18
**Repository:** Budyko-Analysis - Chinese Watershed Hydrological Energy Balance Analysis Framework

This document provides comprehensive guidance for AI assistants (like Claude) working with this codebase.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Critical Concepts](#critical-concepts)
3. [Repository Structure](#repository-structure)
4. [Code Architecture & Conventions](#code-architecture--conventions)
5. [Key Components Reference](#key-components-reference)
6. [Development Workflows](#development-workflows)
7. [Testing Strategy](#testing-strategy)
8. [Documentation](#documentation)
9. [Common Tasks & Patterns](#common-tasks--patterns)
10. [Git Workflow](#git-workflow)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Purpose

This framework analyzes water-energy balance in 6000+ small Chinese watersheds using the Budyko framework. It integrates:

1. **Traditional Budyko Theory** - Water-energy balance at catchment scale
2. **Ibrahim (2025) Deviation Analysis** - Quantifies temporal stability of deviations from theoretical curves
3. **Jaramillo (2022) Trajectory Analysis** - Analyzes movement direction and intensity in Budyko space
4. **He (2023) 3D Framework** - Introduces storage change (ΔS) as third dimension
5. **Core Innovation** - Dynamic PET estimation considering LAI and CO₂ concentration

### Research Context (Framework-WBET)

The project addresses five key research questions:
1. Do watersheds follow Budyko curves?
2. How significant is snow influence?
3. Why do watersheds deviate from Budyko curves?
4. **How does considering LAI and CO₂ improve PET estimates?** (Novel contribution)
5. What other improvements can be made?

### Primary Language

- **Code:** English (variables, functions, classes, comments)
- **Documentation:** Bilingual (Chinese primary, English secondary)
- **User-facing messages:** Chinese

---

## Critical Concepts

### 🔑 THE ABSOLUTE FOUNDATION: Runoff Data (Q)

**CRITICAL: Without runoff data Q, Budyko analysis is IMPOSSIBLE.**

#### Why Q is Essential

Actual evaporation (EA) **cannot be directly measured** at watershed scale. We calculate it using the water balance equation:

```
Water Balance: P - Q = EA + ΔS

Long-term assumption (ΔS ≈ 0):
EA ≈ P - Q

Therefore:
Evaporation Index IE = EA / P = (P - Q) / P
```

**Q determines the Y-axis position in Budyko space. Without Q, we have no "ground truth" for actual water consumption.**

#### Three Roles of Runoff Data

1. **"Ruler" Role** - Measures real watershed water consumption
   ```python
   # Theoretical position (climate-driven)
   IE_theory = f(IA, ω)

   # Actual position (Q-determined!)
   IE_actual = (P - Q) / P

   # Deviation
   deviation = IE_actual - IE_theory
   ```

2. **"Patient" Role** - Reveals watershed "health status"
   - Q decreases → IE increases → Possible irrigation withdrawal
   - Q increases → IE decreases → Possible deforestation

3. **"Reference" Role** - Validates theories and methods
   - Which PET method is best? → Compare with Q-revealed reality
   - Is GRACE data accurate? → Compare P-Q vs P-Q-ΔS

### Budyko Framework Fundamentals

**Budyko Space:**
- **X-axis:** Aridity Index `IA = PET / P`
- **Y-axis:** Evaporation Index `IE = EA / P = (P - Q) / P`

**Budyko Curve (Tixeront-Fu):**
```
IE = 1 + IA - (1 + IA^ω)^(1/ω)
```
where ω is the catchment-specific parameter (typical range: 1.5-4.0)

**Physical Limits:**
- Energy limit: `IE = IA` (slope = 1 line)
- Water limit: `IE = 1` (horizontal line)

### Three Analysis Frameworks

#### 1. Ibrahim (2025) Deviation Analysis

Quantifies **temporal stability** of deviations:

```python
# Step 1: Fit ω for period i
ω_i = fit_omega(IA_i, IE_i)

# Step 2: Calculate deviation for period i+1
IE_theory = budyko_curve(IA_i+1, ω_i)
ε = IE_obs_i+1 - IE_theory

# Step 3: Classify stability
# Stable, Variable, Alternating, Shift
```

#### 2. Jaramillo (2022) Trajectory Analysis

Analyzes **movement** in Budyko space:

```python
# Movement vector
v = (ΔIA, ΔIE)

# Intensity
I = sqrt(ΔIA² + ΔIE²)

# Direction angle
θ = arctan2(ΔIA, ΔIE)

# Follows curve?
follows = (45° < θ < 90°) or (225° < θ < 270°)
```

#### 3. He (2023) 3D Framework

Introduces **storage change** dimension:

```python
# Storage Change Index
SCI = ΔS / P  (from GRACE TWS data)

# Extended evaporation
EA_ext = P - Q - ΔS
IE_ext = EA_ext / P
```

### Innovation: LAI + CO₂ PET Method

**Novel contribution** - Previous work ignored vegetation dynamics and CO₂ effects:

```python
# Traditional Penman-Monteith: static canopy resistance
rs = rs_min

# Innovative approach: dynamic resistance
rs = rs_min * f(LAI) * f(CO2)

where:
  f(LAI) = rs_ref / max(LAI, 0.5)
  f(CO2) = 1 + k_co2 * log(CO2 / CO2_ref)
  k_co2 ≈ 0.15-0.25
```

**Physical mechanisms:**
- LAI increases → More transpiration area → PET increases
- CO₂ increases → Stomata partially close → PET decreases (CO₂ fertilization effect)

---

## Repository Structure

```
Budyko-Analysis/
├── src/                           # CORE CODE (Single Source of Truth)
│   ├── budyko/                    # Budyko analysis core
│   │   ├── water_balance.py       # Q → EA → IE calculations
│   │   ├── curves.py              # Budyko curve formulas
│   │   ├── deviation.py           # Ibrahim (2025) deviation analysis
│   │   └── trajectory_jaramillo.py # Jaramillo (2022) trajectory analysis
│   │
│   ├── models/                    # PET models
│   │   ├── pet_models.py          # Standard PET methods (PM, Hargreaves, etc.)
│   │   └── pet_lai_co2.py         # ⭐ LAI+CO2 PET (INNOVATION)
│   │
│   ├── analysis/                  # High-level analysis
│   │   ├── deviation_attribution.py  # Random forest attribution
│   │   ├── snow_analyzer.py         # Snow influence analysis
│   │   └── budyko_ml_workflow.py    # ML-based workflow
│   │
│   ├── data_processing/           # Data loading (SSOT)
│   │   ├── basin_processor.py     # ⭐ Watershed data (Q loading!)
│   │   ├── grace_lai_processor.py # GRACE TWS & LAI data
│   │   └── cmip6_processor.py     # CMIP6 climate data
│   │
│   ├── visualization/             # Plotting
│   │   ├── budyko_plots.py        # Budyko space plots
│   │   └── direction_rose.py      # Trajectory rose diagrams
│   │
│   └── utils/                     # Utilities
│       ├── parallel_processing.py # Parallel computation for 6000+ basins
│       ├── hydrology.py           # Hydrological utilities
│       └── plotting_config.py     # Matplotlib configuration
│
├── examples/                      # COMPLETE EXAMPLES
│   ├── 01_real_data_workflow.py   # ⭐ FLAGSHIP EXAMPLE (comprehensive)
│   ├── complete_workflow_example.py
│   ├── complex_integrated_analysis.py
│   └── flagship_applicability_analysis.py
│
├── tests/                         # Testing
│   ├── unit/                      # Unit tests
│   │   ├── test_water_balance.py
│   │   ├── test_pet_models.py
│   │   └── test_deviation.py
│   ├── test_complete_workflow.py  # Integration tests
│   └── conftest.py                # pytest fixtures
│
├── scripts/                       # Batch processing
│   ├── bench_parallel_processing.py
│   └── (other scripts)
│
├── docs/                          # Documentation
│   ├── METHODOLOGY.md             # Theoretical background
│   └── 中文字体配置说明.md          # Chinese font setup
│
├── Cheng-3D-Budyko/              # Subproject (3D framework implementation)
│
├── README.md                      # Main documentation (Chinese)
├── QUICKSTART.md                  # Quick start guide (Chinese)
├── 研究思路.md                     # Research rationale (Chinese)
├── 代码库结构.md                   # Original structure doc (Chinese)
├── REFACTOR_SUMMARY.md            # Recent refactoring summary
├── OPTIMIZATION_SUMMARY.md        # Performance optimization summary
├── requirements.txt               # Python dependencies
└── LICENSE                        # MIT License
```

---

## Code Architecture & Conventions

### Design Principles

1. **Single Source of Truth (SSOT)**
   - Each functionality has ONE authoritative implementation
   - No duplicate code across modules
   - Clear import hierarchy

2. **Modularity**
   - Clear separation: data → computation → analysis → visualization
   - Minimal coupling between modules
   - Maximum cohesion within modules

3. **Performance**
   - Vectorized NumPy operations (avoid loops)
   - Parallel processing for batch operations (joblib)
   - Efficient algorithms (e.g., L-BFGS-B for ω fitting)

### Python Style

**Version:** Python 3.8+

**Code Style:**
- Follow PEP 8
- Line length: ~100 characters (flexible)
- Import order: stdlib → third-party → local

**Type Hints:**
```python
from typing import Dict, Optional, Tuple
import numpy as np

def calculate_budyko_indices(
    P: np.ndarray,
    Q: np.ndarray,
    PET: np.ndarray,
    delta_S: Optional[np.ndarray] = None
) -> WaterBalanceResults:
    """Calculate water balance and Budyko indices."""
    ...
```

**Docstrings:**
- Use Google style
- Include Parameters, Returns, Raises, Examples
- Chinese docstrings for module-level docs
- English docstrings for function/class-level docs

```python
def fit_omega(ia_values: np.ndarray, ie_values: np.ndarray) -> Tuple[float, dict]:
    """
    Fit catchment-specific ω parameter.

    Parameters
    ----------
    ia_values : np.ndarray
        Observed aridity index time series
    ie_values : np.ndarray
        Observed evaporation index time series

    Returns
    -------
    omega_opt : float
        Optimal ω parameter
    result : dict
        Fitting statistics

    Examples
    --------
    >>> omega, stats = fit_omega(IA, IE)
    >>> print(f"ω = {omega:.2f}, R² = {stats['r2']:.3f}")
    """
```

**Dataclasses:**
- Use `@dataclass` for result containers
- Provides automatic `__init__`, `__repr__`, etc.

```python
from dataclasses import dataclass

@dataclass
class WaterBalanceResults:
    """Water balance calculation results."""
    precipitation: np.ndarray
    runoff: np.ndarray
    actual_evaporation: np.ndarray
    aridity_index: np.ndarray
    evaporation_index: np.ndarray
    data_quality_flags: np.ndarray
```

### File Organization

**Module Structure:**
```python
# file_name.py
"""Module-level docstring (Chinese OK).

详细说明模块功能和使用方法。
"""
from __future__ import annotations  # For forward references

# Standard library imports
import warnings
from typing import Dict, Optional

# Third-party imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Local imports
from ..utils import some_utility


# Constants
DEFAULT_OMEGA = 2.6

# Classes
class ClassName:
    """Class docstring."""
    ...

# Functions
def function_name():
    """Function docstring."""
    ...
```

### Naming Conventions

**Variables:**
- Scientific: `P`, `Q`, `PET`, `EA`, `IA`, `IE` (uppercase, standard notation)
- Regular: `snake_case`
- Private: `_leading_underscore`

**Functions/Methods:**
- `snake_case`
- Verbs for actions: `calculate_`, `fit_`, `analyze_`

**Classes:**
- `PascalCase`
- Descriptive nouns: `WaterBalanceCalculator`, `BudykoCurves`

**Constants:**
- `UPPER_SNAKE_CASE`

### Dependencies

**Core Scientific Stack:**
```
numpy >= 1.20.0          # Array operations
scipy >= 1.7.0           # Optimization, statistics
pandas >= 1.3.0          # Data manipulation
```

**Machine Learning:**
```
scikit-learn >= 1.0.0    # Random forest, ML utilities
```

**Visualization:**
```
matplotlib >= 3.4.0      # Plotting
seaborn >= 0.11.0        # Statistical plots
```

**Optional (Geospatial):**
```
# Uncomment in requirements.txt if needed
# geopandas >= 0.10.0
# xarray >= 0.19.0
# rasterio >= 1.2.0
# netCDF4 >= 1.5.7
```

**Utilities:**
```
joblib >= 1.0.0          # Parallel processing
tqdm >= 4.62.0           # Progress bars
pytest >= 6.2.0          # Testing
```

---

## Key Components Reference

### 1. Water Balance (`src/budyko/water_balance.py`)

**Primary class:** `WaterBalanceCalculator`

**Purpose:** Calculate EA from Q, compute Budyko indices

**Key method:**
```python
calculator = WaterBalanceCalculator(
    allow_negative_ea=False,
    min_precipitation=100.0,
    max_runoff_ratio=0.95
)

results: WaterBalanceResults = calculator.calculate_budyko_indices(
    P=precipitation,
    Q=runoff,           # CORE INPUT!
    PET=potential_et,
    delta_S=storage_change  # Optional (GRACE data)
)

# Access results
IA = results.aridity_index       # PET / P
IE = results.evaporation_index   # (P - Q) / P
EA = results.actual_evaporation  # P - Q (or P - Q - ΔS)
```

**Returns:** `WaterBalanceResults` dataclass with fields:
- `precipitation`, `runoff`, `pet`, `actual_evaporation`
- `aridity_index`, `evaporation_index`, `runoff_coefficient`
- `data_quality_flags` (for QA/QC)
- Optional: `storage_change`, `closure_error`, etc.

### 2. Budyko Curves (`src/budyko/curves.py`)

**Primary class:** `BudykoCurves`

**Key methods:**

```python
budyko = BudykoCurves()

# Calculate IE from IA and ω
IE_theory = budyko.tixeront_fu(aridity_index=IA, omega=2.6)

# Fit ω parameter from observations
omega, stats = budyko.fit_omega(
    ia_values=observed_IA,
    ie_values=observed_IE,  # From Q!
    initial_omega=2.6
)
# stats contains: 'rmse', 'mae', 'r2', 'n_points'

# Original Budyko (1948) non-parametric curve
IE_budyko1948 = budyko.budyko_1948(aridity_index=IA)
```

**Optimization:** Uses L-BFGS-B for fast convergence (~20 iterations)

### 3. Ibrahim Deviation Analysis (`src/budyko/deviation.py`)

**Primary class:** `DeviationAnalysis`

**Purpose:** Temporal stability analysis of Budyko deviations

```python
analyzer = DeviationAnalysis(period_length=20)

# Calculate deviation distribution between periods
distribution = analyzer.calculate_deviations(
    ia_i=period1_aridity,
    ie_obs_i=period1_evaporation,      # From Q
    omega_i=period1_omega,
    ia_i_plus_1=period2_aridity,
    ie_obs_i_plus_1=period2_evaporation,  # From Q
    period_pair='Δ1-2'
)

# Statistical test
test_result = analyzer.wilcoxon_test(distribution)

# Classify temporal stability
# Returns: 'Stable', 'Variable', 'Alternating', 'Shift'
stability_class = analyzer.classify_temporal_pattern(
    distributions=[dist1, dist2, dist3]
)
```

### 4. Jaramillo Trajectory Analysis (`src/budyko/trajectory_jaramillo.py`)

**Primary class:** `TrajectoryAnalyzer`

**Purpose:** Analyze movement in Budyko space

```python
trajectory = TrajectoryAnalyzer()

movement = trajectory.calculate_movement(
    catchment_id='basin_001',
    period_1={'IA': ia1, 'IE': ie1, 'name': '1980-2000'},
    period_2={'IA': ia2, 'IE': ie2, 'name': '2000-2020'}
)

# Access movement metrics
print(f"Intensity: {movement.intensity:.3f}")
print(f"Direction: {movement.direction_angle:.1f}°")
print(f"Follows curve: {movement.follows_curve}")
print(f"Movement type: {movement.movement_type}")
```

**Movement classification:**
- Along curve: `45° < θ < 90°` or `225° < θ < 270°`
- Perpendicular to curve: Other angles

### 5. PET Models (`src/models/pet_models.py`)

**Factory pattern:** `PETModelFactory`

**Available methods:**
```python
from src.models.pet_models import PETModelFactory

# Standard Penman-Monteith (FAO-56)
pet_pm = PETModelFactory.create('penman_monteith')
PET_pm = pet_pm.calculate(
    temperature=T,
    humidity=RH,
    wind_speed=u2,
    radiation=Rn,
    latitude=lat
)

# Hargreaves (temperature-only)
pet_hs = PETModelFactory.create('hargreaves')
PET_hs = pet_hs.calculate(
    temperature_max=Tmax,
    temperature_min=Tmin,
    latitude=lat
)

# Priestley-Taylor
pet_pt = PETModelFactory.create('priestley_taylor')
PET_pt = pet_pt.calculate(
    temperature=T,
    radiation=Rn,
    alpha=1.26  # PT coefficient
)
```

### 6. ⭐ LAI + CO₂ PET Model (`src/models/pet_lai_co2.py`)

**Primary class:** `PETWithLAICO2`

**Innovation:** Dynamic stomatal resistance considering vegetation and CO₂

```python
from src.models.pet_lai_co2 import PETWithLAICO2

pet_advanced = PETWithLAICO2(
    k_co2=0.20,           # CO2 sensitivity (0.15-0.25)
    co2_reference=380.0,  # Reference CO2 (ppm)
    rs_min=100.0          # Minimum stomatal resistance (s/m)
)

PET = pet_advanced.calculate(
    temperature=T,
    humidity=RH,
    wind_speed=u2,
    radiation=Rn,
    lai=LAI,              # MODIS LAI
    co2=CO2,              # Atmospheric CO2 (ppm)
    latitude=lat
)

# Compare with standard method
PET_standard = pet_pm.calculate(...)
difference = PET - PET_standard  # Typically PET is lower due to CO2 effect
```

### 7. Deviation Attribution (`src/analysis/deviation_attribution.py`)

**Primary class:** `DeviationAttribution`

**Purpose:** Machine learning-based attribution of Budyko deviations

```python
from src.analysis.deviation_attribution import DeviationAttribution

attributor = DeviationAttribution()

# Add driver variables
attributor.add_drivers({
    'land_use_change': land_use_data,
    'irrigation': irrigation_data,
    'reservoir': reservoir_data,
    'snow_fraction': snow_data,
    'lai': lai_data,
    'co2': co2_data
})

# Perform attribution (Random Forest)
results = attributor.attribute_deviation(
    deviation=budyko_deviation,  # From Q observations
    method='random_forest'
)

# Feature importance
print(results['importance'])
# Example output:
# {'irrigation': 0.35, 'lai': 0.28, 'reservoir': 0.18, ...}
```

### 8. Parallel Processing (`src/utils/parallel_processing.py`)

**Primary class:** `ParallelBudykoAnalyzer`

**Purpose:** Efficient batch processing for 6000+ watersheds

```python
from src.utils.parallel_processing import ParallelBudykoAnalyzer

analyzer = ParallelBudykoAnalyzer(n_jobs=-1)  # Use all CPUs

# Process thousands of catchments
results_all = analyzer.analyze_catchments(
    catchment_data=basin_data,
    pet_method='lai_co2',  # Use innovative PET
    n_catchments=6000
)
```

### 9. Visualization (`src/visualization/budyko_plots.py`)

**Primary class:** `BudykoPlotter`

```python
from src.visualization.budyko_plots import BudykoPlotter

plotter = BudykoPlotter()

# Plot Budyko space
fig, ax = plotter.plot_budyko_space(
    aridity_index=IA,
    evaporation_index=IE,
    omega=2.6,
    color_by='deviation',
    size_by='basin_area'
)

# Add theoretical curves
plotter.add_budyko_curves(ax, omega_range=[1.5, 2.0, 2.5, 3.0])
plotter.add_water_energy_limits(ax)

plt.savefig('budyko_space.png', dpi=300)
```

---

## Development Workflows

### Adding New Features

**General pattern:**

1. **Identify the appropriate module**
   - PET calculation → `src/models/`
   - Budyko analysis → `src/budyko/`
   - High-level analysis → `src/analysis/`
   - Data loading → `src/data_processing/`

2. **Check for existing implementations** (avoid duplication)

3. **Write the feature**
   - Add type hints
   - Write docstrings (Google style)
   - Use vectorized operations (NumPy)
   - Handle edge cases (NaN, zero, negative values)

4. **Add unit tests** (`tests/unit/`)
   ```python
   # tests/unit/test_new_feature.py
   import pytest
   import numpy as np
   from src.module.new_feature import NewClass

   def test_new_feature_basic():
       obj = NewClass()
       result = obj.calculate(input_data)
       assert np.allclose(result, expected_value)

   def test_new_feature_edge_cases():
       obj = NewClass()
       with pytest.raises(ValueError):
           obj.calculate(invalid_input)
   ```

5. **Add example usage** (`examples/`)

6. **Update documentation**
   - Add to relevant `.md` file
   - Update this CLAUDE.md if needed

### Adding New PET Methods

**Pattern:**

```python
# src/models/pet_models.py

class NewPETMethod(PETModelBase):
    """New PET calculation method.

    References
    ----------
    Author et al. (Year). Journal.
    """

    def __init__(self, **params):
        super().__init__(name="new_method")
        self.params = params

    def calculate(self, **inputs) -> np.ndarray:
        """
        Calculate PET using new method.

        Parameters
        ----------
        inputs : dict
            Required meteorological inputs

        Returns
        -------
        np.ndarray
            PET values (mm/day)
        """
        # Vectorized calculation
        PET = ...
        return PET

    def validate_inputs(self, **inputs):
        """Validate required inputs."""
        required = ['temperature', 'radiation']
        for key in required:
            if key not in inputs:
                raise ValueError(f"Missing required input: {key}")

# Register in factory
PETModelFactory.register('new_method', NewPETMethod)
```

### Performance Optimization

**When optimizing:**

1. **Profile first**
   ```bash
   python -m cProfile -o profile.stats script.py
   python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
   ```

2. **Vectorize loops**
   ```python
   # ❌ Slow
   result = []
   for i in range(len(data)):
       result.append(func(data[i]))

   # ✅ Fast
   result = func(data)  # Vectorized
   ```

3. **Use NumPy efficiently**
   ```python
   # ❌ Avoid creating intermediate arrays
   result = ((a + b) * c) / d

   # ✅ Use in-place operations if possible
   result = a.copy()
   result += b
   result *= c
   result /= d
   ```

4. **Parallel processing for batch operations**
   ```python
   from joblib import Parallel, delayed

   results = Parallel(n_jobs=-1)(
       delayed(process_basin)(basin_id)
       for basin_id in basin_ids
   )
   ```

5. **Document optimizations**
   - Add comments explaining performance-critical code
   - Update `PERFORMANCE_OPTIMIZATIONS.md`

---

## Testing Strategy

### Test Structure

```
tests/
├── unit/                  # Unit tests (fast, isolated)
│   ├── test_water_balance.py
│   ├── test_pet_models.py
│   ├── test_deviation.py
│   └── test_curves.py
├── integration/           # Integration tests (slower, end-to-end)
│   └── test_full_workflow.py
├── test_complete_workflow.py
└── conftest.py            # pytest fixtures
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/unit/test_water_balance.py -v

# Specific test function
pytest tests/unit/test_water_balance.py::test_basic_calculation -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Fast tests only (exclude slow integration tests)
pytest tests/unit/ -v
```

### Writing Tests

**Unit test template:**

```python
import pytest
import numpy as np
from src.budyko.water_balance import WaterBalanceCalculator

class TestWaterBalanceCalculator:
    """Test suite for WaterBalanceCalculator."""

    def test_basic_calculation(self):
        """Test basic water balance calculation."""
        calculator = WaterBalanceCalculator()

        P = np.array([800, 850, 900])
        Q = np.array([200, 220, 240])
        PET = np.array([1000, 1050, 1100])

        results = calculator.calculate_budyko_indices(P, Q, PET)

        # Test EA = P - Q
        expected_EA = P - Q
        np.testing.assert_array_almost_equal(
            results.actual_evaporation,
            expected_EA
        )

        # Test IE = EA / P
        expected_IE = expected_EA / P
        np.testing.assert_array_almost_equal(
            results.evaporation_index,
            expected_IE
        )

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        calculator = WaterBalanceCalculator()

        # Mismatched lengths
        with pytest.raises(ValueError):
            calculator.calculate_budyko_indices(
                P=np.array([800]),
                Q=np.array([200, 220]),
                PET=np.array([1000])
            )

    def test_edge_cases(self):
        """Test edge cases."""
        calculator = WaterBalanceCalculator(allow_negative_ea=False)

        # Q > P (negative EA)
        P = np.array([100])
        Q = np.array([150])  # Q > P
        PET = np.array([200])

        results = calculator.calculate_budyko_indices(P, Q, PET)

        # Should clip to 0
        assert results.actual_evaporation[0] >= 0
```

### Test Fixtures (conftest.py)

```python
import pytest
import numpy as np

@pytest.fixture
def sample_basin_data():
    """Provide sample basin data for tests."""
    return {
        'P': np.random.uniform(500, 1500, 20),
        'Q': np.random.uniform(100, 500, 20),
        'PET': np.random.uniform(800, 1800, 20),
        'T': np.random.uniform(10, 25, 20),
        'LAI': np.random.uniform(2.0, 5.0, 20),
        'CO2': np.linspace(350, 420, 20)
    }
```

---

## Documentation

### Documentation Files

**English:**
- `CLAUDE.md` (this file) - AI assistant guide
- `QUICKSTART.md` - Quick start (bilingual)
- `docs/METHODOLOGY.md` - Theoretical background

**Chinese:**
- `README.md` - Main documentation
- `研究思路.md` - Research rationale
- `代码库结构.md` - Repository structure
- `REFACTOR_SUMMARY.md` - Refactoring summary
- `OPTIMIZATION_SUMMARY.md` - Performance optimizations

### Documentation Standards

**When documenting:**

1. **Function/Class docstrings** - English, Google style
2. **Module docstrings** - Chinese or English
3. **Comments** - English preferred
4. **User-facing docs** - Chinese primary, English secondary
5. **Examples** - Include working code with expected output

**Docstring example:**

```python
def calculate_budyko_indices(
    P: np.ndarray,
    Q: np.ndarray,
    PET: np.ndarray,
    delta_S: Optional[np.ndarray] = None
) -> WaterBalanceResults:
    """
    Calculate water balance and Budyko indices from observations.

    This function computes actual evaporation (EA) from runoff observations
    using the water balance equation, then calculates key Budyko indices.

    Parameters
    ----------
    P : np.ndarray
        Precipitation (mm)
    Q : np.ndarray
        Runoff (mm) - CORE INPUT for Budyko analysis
    PET : np.ndarray
        Potential evapotranspiration (mm)
    delta_S : np.ndarray, optional
        Storage change (mm) from GRACE TWS data

    Returns
    -------
    WaterBalanceResults
        Dataclass containing:
        - actual_evaporation: EA = P - Q (or P - Q - ΔS)
        - aridity_index: IA = PET / P
        - evaporation_index: IE = EA / P
        - runoff_coefficient: RC = Q / P
        - data_quality_flags: QA/QC flags

    Raises
    ------
    ValueError
        If input arrays have different lengths

    Notes
    -----
    The water balance equation is:
        P - Q = EA + ΔS

    On long time scales (annual, multi-year), we assume ΔS ≈ 0:
        EA ≈ P - Q

    References
    ----------
    Budyko, M.I. (1974). Climate and Life. Academic Press.

    Examples
    --------
    >>> calculator = WaterBalanceCalculator()
    >>> results = calculator.calculate_budyko_indices(P, Q, PET)
    >>> print(f"Mean IE: {np.mean(results.evaporation_index):.2f}")
    Mean IE: 0.75
    """
```

---

## Common Tasks & Patterns

### Task 1: Basic Budyko Analysis

```python
import numpy as np
from src.budyko.water_balance import WaterBalanceCalculator
from src.budyko.curves import BudykoCurves

# 1. Prepare data
P = np.array([800, 850, 900, 950, 1000])   # Precipitation
Q = np.array([200, 220, 240, 260, 280])     # RUNOFF - ESSENTIAL!
PET = np.array([1000, 1050, 1100, 1150, 1200])  # Potential ET

# 2. Calculate water balance
wb_calc = WaterBalanceCalculator()
results = wb_calc.calculate_budyko_indices(P, Q, PET)

# 3. Fit Budyko curve
budyko = BudykoCurves()
omega, stats = budyko.fit_omega(
    results.aridity_index,
    results.evaporation_index
)

print(f"ω parameter: {omega:.2f}")
print(f"R²: {stats['r2']:.3f}")

# 4. Calculate deviation
IE_theory = budyko.tixeront_fu(results.aridity_index, omega)
deviation = results.evaporation_index - IE_theory
print(f"Mean deviation: {np.mean(deviation):.3f}")
```

### Task 2: Compare PET Methods

```python
from src.models.pet_models import PETModelFactory
from src.models.pet_lai_co2 import PETWithLAICO2

# Meteorological inputs
T = np.array([15, 16, 14, 15, 16])
RH = np.array([60, 65, 70, 62, 68])
u2 = np.array([2.0, 2.5, 1.8, 2.2, 2.0])
Rn = np.array([200, 210, 190, 205, 200])
lat = 30.0  # degrees

# Standard method
pet_pm = PETModelFactory.create('penman_monteith')
PET_standard = pet_pm.calculate(
    temperature=T,
    humidity=RH,
    wind_speed=u2,
    radiation=Rn,
    latitude=lat
)

# Innovative method (with LAI and CO2)
LAI = np.array([3.5, 3.8, 3.2, 3.6, 3.9])
CO2 = np.array([380, 390, 400, 410, 420])

pet_advanced = PETWithLAICO2()
PET_innovative = pet_advanced.calculate(
    temperature=T,
    humidity=RH,
    wind_speed=u2,
    radiation=Rn,
    lai=LAI,
    co2=CO2,
    latitude=lat
)

# Compare
difference = PET_innovative - PET_standard
print(f"Mean PET difference: {np.mean(difference):.2f} mm")
print(f"Relative difference: {np.mean(difference / PET_standard) * 100:.1f}%")
```

### Task 3: Ibrahim Deviation Analysis

```python
from src.budyko.deviation import DeviationAnalysis
from src.budyko.curves import BudykoCurves

# Divide data into periods
period1_years = range(1980, 2000)
period2_years = range(2000, 2020)

# Calculate indices for each period
IA_p1 = ...  # From P and PET
IE_p1 = ...  # From P and Q
IA_p2 = ...
IE_p2 = ...

# Fit omega for period 1
budyko = BudykoCurves()
omega_p1, _ = budyko.fit_omega(IA_p1, IE_p1)

# Calculate deviation for period 2
analyzer = DeviationAnalysis()
distribution = analyzer.calculate_deviations(
    ia_i=IA_p1,
    ie_obs_i=IE_p1,
    omega_i=omega_p1,
    ia_i_plus_1=IA_p2,
    ie_obs_i_plus_1=IE_p2,
    period_pair='1980-2000 → 2000-2020'
)

# Statistical test
test_result = analyzer.wilcoxon_test(distribution)
print(f"p-value: {test_result['p_value']:.4f}")
print(f"Significant: {test_result['significant']}")
```

### Task 4: Jaramillo Trajectory Analysis

```python
from src.budyko.trajectory_jaramillo import TrajectoryAnalyzer

trajectory = TrajectoryAnalyzer()

# Define two periods
period1 = {
    'IA': np.mean(IA_1980_2000),
    'IE': np.mean(IE_1980_2000),
    'name': '1980-2000'
}

period2 = {
    'IA': np.mean(IA_2000_2020),
    'IE': np.mean(IE_2000_2020),
    'name': '2000-2020'
}

# Calculate movement
movement = trajectory.calculate_movement(
    catchment_id='basin_001',
    period_1=period1,
    period_2=period2
)

print(f"Movement intensity: {movement.intensity:.3f}")
print(f"Direction angle: {movement.direction_angle:.1f}°")
print(f"Follows Budyko curve: {movement.follows_curve}")
print(f"Movement type: {movement.movement_type}")
```

### Task 5: Batch Processing Multiple Watersheds

```python
from src.utils.parallel_processing import ParallelBudykoAnalyzer
from joblib import Parallel, delayed

def process_single_basin(basin_id, basin_data):
    """Process single watershed."""
    # Extract basin data
    P = basin_data[basin_id]['P']
    Q = basin_data[basin_id]['Q']
    PET = basin_data[basin_id]['PET']

    # Calculate water balance
    wb_calc = WaterBalanceCalculator()
    results = wb_calc.calculate_budyko_indices(P, Q, PET)

    # Fit omega
    budyko = BudykoCurves()
    omega, stats = budyko.fit_omega(
        results.aridity_index,
        results.evaporation_index
    )

    return {
        'basin_id': basin_id,
        'omega': omega,
        'r2': stats['r2'],
        'mean_IA': np.mean(results.aridity_index),
        'mean_IE': np.mean(results.evaporation_index)
    }

# Parallel processing
basin_ids = list(basin_data.keys())
results = Parallel(n_jobs=-1, verbose=1)(
    delayed(process_single_basin)(basin_id, basin_data)
    for basin_id in basin_ids
)

# Convert to DataFrame
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df.describe())
```

### Task 6: Plotting Results

```python
import matplotlib.pyplot as plt
from src.visualization.budyko_plots import BudykoPlotter

# Create plotter
plotter = BudykoPlotter()

# Plot Budyko space
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot with coloring
scatter = ax.scatter(
    IA,
    IE,
    c=deviation,  # Color by deviation
    s=50,
    cmap='RdBu_r',
    vmin=-0.2,
    vmax=0.2,
    alpha=0.7
)

# Add Budyko curves
for omega_val in [1.5, 2.0, 2.5, 3.0]:
    ia_range = np.linspace(0, 5, 100)
    ie_curve = budyko.tixeront_fu(ia_range, omega_val)
    ax.plot(ia_range, ie_curve, 'k--', alpha=0.3, label=f'ω={omega_val}')

# Add limits
ax.plot([0, 5], [0, 5], 'k-', alpha=0.3, label='Energy limit')
ax.axhline(1, color='k', linestyle='-', alpha=0.3, label='Water limit')

# Formatting
ax.set_xlabel('Aridity Index (IA = PET/P)', fontsize=12)
ax.set_ylabel('Evaporation Index (IE = EA/P)', fontsize=12)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.2)
ax.legend()
ax.grid(True, alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Budyko Deviation', fontsize=12)

plt.tight_layout()
plt.savefig('budyko_space.png', dpi=300, bbox_inches='tight')
```

---

## Git Workflow

### Branch Structure

**Main branches:**
- `main` / `master` - Stable release branch
- `claude/*` - AI assistant working branches (auto-generated)

**Feature branches:**
- `feature/feature-name` - New features
- `fix/bug-name` - Bug fixes
- `docs/update-name` - Documentation updates
- `perf/optimization-name` - Performance improvements

### Commit Conventions

**Format:**
```
<type>: <subject>

<body (optional)>

<footer (optional)>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Formatting
- `refactor` - Code restructuring
- `perf` - Performance improvement
- `test` - Tests
- `chore` - Maintenance

**Examples:**
```bash
git commit -m "feat: Add LAI+CO2 PET model with stomatal resistance adjustment"

git commit -m "fix: Handle negative EA values in water balance calculation"

git commit -m "docs: Update CLAUDE.md with trajectory analysis examples"

git commit -m "perf: Vectorize omega fitting loop (30% speed improvement)"
```

### Working with Claude Branches

**Claude branches follow pattern:** `claude/claude-md-<session-id>`

```bash
# Check current branch
git branch

# Create and switch to Claude branch (if needed)
git checkout -b claude/claude-md-new-session-id

# Stage changes
git add src/module/file.py

# Commit
git commit -m "feat: Add new functionality"

# Push (use -u for first push)
git push -u origin claude/claude-md-session-id

# Subsequent pushes
git push origin claude/claude-md-session-id
```

### Pull Request Process

1. **Ensure tests pass**
   ```bash
   pytest tests/ -v
   ```

2. **Push to feature branch**
   ```bash
   git push -u origin feature/feature-name
   ```

3. **Create PR** (via GitHub UI or `gh` CLI)
   ```bash
   gh pr create --title "Add new feature" --body "Description of changes"
   ```

4. **PR checklist:**
   - [ ] Tests pass
   - [ ] Documentation updated
   - [ ] Code follows style guide
   - [ ] No merge conflicts
   - [ ] Descriptive commit messages

---

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors

**Problem:**
```python
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Ensure you're in repository root
cd /path/to/Budyko-Analysis

# Run Python with module flag
python -m src.module.file

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Budyko-Analysis"
```

#### Issue 2: Negative EA Values

**Problem:**
```
Warning: Negative actual evaporation detected (Q > P)
```

**Solution:**
```python
# Option 1: Clip to zero
calculator = WaterBalanceCalculator(allow_negative_ea=False)

# Option 2: Allow negative (for data quality checking)
calculator = WaterBalanceCalculator(allow_negative_ea=True)
results = calculator.calculate_budyko_indices(P, Q, PET)

# Investigate negative EA cases
negative_mask = results.actual_evaporation < 0
print(f"Negative EA in {negative_mask.sum()} / {len(P)} cases")
```

#### Issue 3: Omega Fitting Fails

**Problem:**
```
RuntimeWarning: Optimal omega not found (R² < 0.5)
```

**Solution:**
```python
# Check data quality
print(f"IA range: {IA.min():.2f} - {IA.max():.2f}")
print(f"IE range: {IE.min():.2f} - {IE.max():.2f}")

# Check for outliers
plt.scatter(IA, IE)
plt.xlabel('IA')
plt.ylabel('IE')
plt.show()

# Try different initial omega
omega, stats = budyko.fit_omega(
    IA, IE,
    initial_omega=2.0  # Try different starting point
)

# Filter problematic data points
valid_mask = (IA > 0) & (IE > 0) & (IE < 1.5)
omega, stats = budyko.fit_omega(IA[valid_mask], IE[valid_mask])
```

#### Issue 4: Performance Bottlenecks

**Problem:** Slow execution with large datasets

**Solution:**
```python
# Profile code
python -m cProfile -o profile.stats script.py

# Analyze
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)

# Use parallel processing
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1)(
    delayed(func)(data_chunk)
    for data_chunk in data_chunks
)

# Vectorize operations
# ❌ Slow
result = [func(x) for x in data]

# ✅ Fast
result = func(data)  # Vectorized NumPy
```

#### Issue 5: Chinese Font Display Issues

**Problem:** Chinese characters show as squares in plots

**Solution:**
```python
# See docs/中文字体配置说明.md for details

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set Chinese font
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# Or use custom config
from src.utils.plotting_config import setup_chinese_font
setup_chinese_font()
```

### Debugging Tips

1. **Use print debugging strategically**
   ```python
   print(f"DEBUG: P shape={P.shape}, Q shape={Q.shape}")
   print(f"DEBUG: IA range=[{IA.min():.2f}, {IA.max():.2f}]")
   ```

2. **Check intermediate results**
   ```python
   # Water balance closure
   closure_error = P - Q - EA - delta_S
   print(f"Max closure error: {np.abs(closure_error).max():.2f} mm")
   ```

3. **Visualize data**
   ```python
   # Quick diagnostic plot
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   axes[0, 0].plot(P)
   axes[0, 0].set_title('Precipitation')
   axes[0, 1].plot(Q)
   axes[0, 1].set_title('Runoff')
   axes[1, 0].plot(EA)
   axes[1, 0].set_title('Actual Evaporation')
   axes[1, 1].scatter(IA, IE)
   axes[1, 1].set_title('Budyko Space')
   plt.tight_layout()
   plt.show()
   ```

4. **Use assertions for data validation**
   ```python
   assert len(P) == len(Q), "P and Q must have same length"
   assert np.all(P >= 0), "Precipitation cannot be negative"
   assert np.all(Q >= 0), "Runoff cannot be negative"
   ```

---

## Key Reminders for AI Assistants

### Critical Points to Remember

1. **🔑 RUNOFF DATA Q IS ESSENTIAL**
   - Without Q, Budyko analysis cannot proceed
   - Q determines EA through water balance: EA = P - Q
   - Q-derived IE is the "ground truth" for Y-axis position
   - Always emphasize Q's importance when discussing water balance

2. **Single Source of Truth (SSOT)**
   - Never duplicate functionality
   - Check existing implementations before creating new ones
   - Use factory patterns (`PETModelFactory`) for extensibility

3. **Performance Matters**
   - This framework handles 6000+ watersheds
   - Always vectorize (use NumPy arrays, not loops)
   - Use parallel processing (`joblib`) for batch operations
   - Document performance optimizations

4. **Documentation is Bilingual**
   - Code: English
   - User docs: Chinese primary
   - Scientific terms: Use standard notation (P, Q, PET, EA, IA, IE, ω)

5. **Three Analysis Frameworks**
   - Ibrahim (2025): Temporal deviation stability
   - Jaramillo (2022): Trajectory movement analysis
   - He (2023): 3D framework with storage change
   - These are NOT mutually exclusive; they complement each other

6. **Innovation: LAI + CO₂ PET**
   - This is the project's core scientific contribution
   - Considers vegetation dynamics (LAI) and CO₂ fertilization effect
   - Located in `src/models/pet_lai_co2.py`

7. **Testing is Required**
   - Add unit tests for new features
   - Use pytest fixtures for common test data
   - Integration tests for workflows

8. **Flagship Example**
   - `examples/01_real_data_workflow.py` is the comprehensive reference
   - Shows all three frameworks + innovation PET + attribution
   - Use it as template for new examples

### When Making Changes

**Always ask yourself:**

1. Does this change affect runoff data (Q) handling? → Extra care needed
2. Is there already an implementation of this? → Check SSOT
3. Is this vectorized? → Performance critical
4. Are there tests? → Add them
5. Is documentation updated? → Update relevant .md files
6. Does this break backward compatibility? → Discuss with user

### Communication Style

**When explaining concepts:**
- Use scientific notation (P, Q, PET, EA, IA, IE)
- Emphasize Q's central role
- Reference line numbers when discussing code
- Provide working code examples
- Explain *why* (physical meaning, mathematical reasoning)

**When implementing features:**
- Follow existing patterns
- Add type hints
- Write docstrings
- Include examples in docstrings
- Handle edge cases (NaN, negative, zero)

---

## Quick Reference

### Key Variables & Notation

| Symbol | Name | Unit | Description |
|--------|------|------|-------------|
| P | Precipitation | mm | Measured or gridded data |
| Q | Runoff | mm | **CORE: Determines EA** |
| PET | Potential ET | mm | Calculated from meteorology |
| EA | Actual Evaporation | mm | **P - Q** (or P - Q - ΔS) |
| ΔS | Storage Change | mm | From GRACE TWS |
| IA | Aridity Index | - | **PET / P** |
| IE | Evaporation Index | - | **EA / P = (P - Q) / P** |
| RC | Runoff Coefficient | - | Q / P |
| ω | Catchment Parameter | - | Budyko curve shape (1.5-4.0) |
| LAI | Leaf Area Index | m²/m² | MODIS data |
| CO₂ | CO₂ Concentration | ppm | Atmospheric measurement |

### Key Equations

**Water Balance:**
```
P - Q = EA + ΔS
EA ≈ P - Q  (long-term)
```

**Budyko Indices:**
```
IA = PET / P
IE = EA / P = (P - Q) / P
```

**Budyko Curve (Tixeront-Fu):**
```
IE = 1 + IA - (1 + IA^ω)^(1/ω)
```

**Ibrahim Deviation:**
```
ε = IE_obs - IE_theory(ω_previous_period)
```

**Jaramillo Movement:**
```
Intensity: I = sqrt(ΔIA² + ΔIE²)
Direction: θ = arctan2(ΔIA, ΔIE)
```

### File Paths Quick Reference

```python
# Core modules
from src.budyko.water_balance import WaterBalanceCalculator
from src.budyko.curves import BudykoCurves
from src.budyko.deviation import DeviationAnalysis
from src.budyko.trajectory_jaramillo import TrajectoryAnalyzer

# PET models
from src.models.pet_models import PETModelFactory
from src.models.pet_lai_co2 import PETWithLAICO2

# Analysis
from src.analysis.deviation_attribution import DeviationAttribution
from src.analysis.snow_analyzer import SnowAnalyzer

# Data processing
from src.data_processing.basin_processor import BasinDataProcessor
from src.data_processing.grace_lai_processor import GRACELAIProcessor

# Visualization
from src.visualization.budyko_plots import BudykoPlotter
from src.visualization.direction_rose import DirectionRosePlotter

# Utilities
from src.utils.parallel_processing import ParallelBudykoAnalyzer
```

---

## Changelog

### 2025-11-18
- Initial CLAUDE.md creation
- Comprehensive documentation of codebase structure
- Added critical concepts section emphasizing Q's importance
- Documented three analysis frameworks
- Included common tasks and patterns
- Added troubleshooting guide

---

## References

### Key Papers

1. **Ibrahim et al. (2025)** - "On the Need to Update the Water-Energy Balance Framework"
   - Deviation analysis methodology
   - Temporal stability classification

2. **Jaramillo et al. (2022)** - "Fewer Basins Will Follow Their Budyko Curves Under Global Warming"
   - Trajectory analysis in Budyko space
   - Movement direction and intensity metrics

3. **He et al. (2023)** - "3D Budyko Framework"
   - Storage change as third dimension
   - GRACE TWS integration

4. **Budyko, M.I. (1974)** - "Climate and Life"
   - Original Budyko hypothesis
   - Water-energy balance theory

5. **Fu, B.P. (1981)** - "On the calculation of the evaporation from land surface"
   - Tixeront-Fu parametric equation
   - ω parameter concept

### Data Sources

- **CMFD** - China Meteorological Forcing Dataset (0.1°, 1960-2020)
- **Caravan** - Global large-sample hydrology dataset
- **MODIS** - MOD15A2H LAI product
- **GRACE** - Terrestrial water storage
- **Mauna Loa** - Atmospheric CO₂ measurements
- **CMIP6** - Climate model projections
- **TRENDY** - Terrestrial ecosystem models

---

## Contact & Support

- **Issues**: GitHub Issues
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Tests**: See `tests/` directory

---

**End of CLAUDE.md**
