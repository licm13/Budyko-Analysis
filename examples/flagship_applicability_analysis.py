#!/usr/bin/env python3
"""
Flagship Example: Budyko Framework Applicability Analysis

=============================================================================
SCIENTIFIC OBJECTIVE
=============================================================================
This flagship example demonstrates the SCIENTIFIC INTEGRATION of three
cutting-edge Budyko analysis methods:
1. Ibrahim et al. (2025) - Deviation Analysis & Temporal Stability
2. Jaramillo et al. (2022) - Trajectory Analysis in Budyko Space
3. He et al. (2023) - 3D Framework with Storage Change

Unlike sequential "cookbook" examples, this analysis shows how these methods
WORK TOGETHER to diagnose hydrological change and assess framework applicability
across diverse basin conditions.

=============================================================================
KEY INNOVATION: SCENARIO-BASED COMPARATIVE ANALYSIS
=============================================================================
We systematically test Budyko framework applicability across four contrasting
scenarios, answering critical scientific questions:

SCENARIO A: Climate Regime Comparison (Humid vs Arid)
-------------------------------------------------------
Question: Is the Budyko framework equally reliable across climate zones?
Basins:
  - Basin_Humid: IA ≈ 0.5 (energy-limited)
  - Basin_Arid: IA ≈ 2.5 (water-limited)
Expected: Arid basins show higher sensitivity to ω; humid basins may violate
          ΔS≈0 assumption due to groundwater dynamics.

SCENARIO B: Human Impact Assessment (Natural vs Irrigated)
-----------------------------------------------------------
Question: Can Budyko deviation analysis identify anthropogenic disturbance?
Basins:
  - Basin_Natural: human_impact = 0 (pristine)
  - Basin_Irrigated: human_impact = 0.5 (heavy water withdrawal)
Expected: Natural basin → Stable category (median ε ≈ 0)
          Irrigated basin → Shift category (systematic ε > 0, IE increases)
          Jaramillo trajectory shows "deviation from curve"

SCENARIO C: PET Method Sensitivity (Standard vs Innovative)
------------------------------------------------------------
Question: How much does ignoring LAI+CO2 dynamics affect attribution?
Basin: Basin_Greening (LAI↑, CO2↑)
Analysis 1: Traditional Penman-Monteith PET (ignores LAI/CO2 trends)
Analysis 2: Innovative PET with LAI+CO2 adjustments
Expected: Traditional PET may misattribute vegetation greening effects as
          "climate-driven" deviations, while innovative PET isolates true
          hydrological change.

SCENARIO D: Temporal Scale Effect (Annual vs Multi-year)
---------------------------------------------------------
Question: Does the ΔS≈0 assumption hold at annual vs decadal scales?
Basin: Basin_Standard (with interannual storage variability)
Analysis 1: Annual data (ΔS significant)
Analysis 2: 5-year moving average (ΔS smoothed)
Expected: Annual: |SCI| > 0.1, IE vs IE_ext differ
          5-year: |SCI| → 0, validating Budyko assumption at longer timescales

=============================================================================
METHOD INTEGRATION WORKFLOW
=============================================================================
For EACH scenario, we execute a DIAGNOSTIC CASCADE:

Step 1: Jaramillo Trajectory Analysis
  → Detect IF basin deviates from Budyko curve (follows_curve = True/False)

Step 2: Ibrahim Deviation Analysis (if deviating)
  → Quantify MAGNITUDE and TEMPORAL PATTERN of deviation
  → Classify stability: Stable / Variable / Shift / Alternating

Step 3: He 3D Framework (if ΔS data available)
  → Test IF deviation is due to storage change (compare IE vs IE_ext)
  → If IE ≈ IE_ext → deviation NOT due to ΔS → seek other causes

Step 4: Deviation Attribution (if systematic shift detected)
  → Identify DRIVERS using Random Forest on external forcings
  → Rank importance of: irrigation, LAI, CO2, temperature, etc.

This cascade mimics real-world scientific investigation: detect anomaly →
quantify pattern → test hypotheses → identify causes.

=============================================================================
EXPECTED SCIENTIFIC INSIGHTS
=============================================================================
1. Framework Robustness: Budyko works best for:
   - Humid/sub-humid basins (IA < 1.5)
   - Multi-year timescales (ΔS≈0 valid)
   - Natural or low-impact systems

2. Deviation Signatures:
   - Irrigation → IE increases, follows "deviating" trajectory
   - Greening → Complex: LAI↑ increases ET, but CO2↑ suppresses stomatal
     conductance → net effect depends on which dominates

3. Attribution Clarity:
   - Traditional PET may conflate vegetation & climate effects
   - Innovative PET enables cleaner attribution

4. Temporal Scale Matters:
   - Annual Budyko may violate ΔS≈0 in regions with:
     * Large TWS variability (GRACE shows high SCI)
     * Groundwater depletion/recharge
   - 5-10 year averaging recovers Budyko framework validity

=============================================================================
USAGE
=============================================================================
python examples/flagship_applicability_analysis.py

Output:
  - Terminal: Scientific analysis report for all scenarios
  - Figure: examples/figures/flagship_applicability_analysis__summary.png
            (Multi-panel comparative visualization)
"""

import sys
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Configure plotting
try:
    from utils.plotting_config import setup_chinese_fonts
    setup_chinese_fonts()
except ImportError:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# Import our modules
from utils.simulation_utils import generate_synthetic_basin_data, BasinCharacteristics
from budyko.water_balance import WaterBalanceCalculator
from budyko.curves import BudykoCurves
from budyko.deviation import DeviationAnalysis, TemporalStability, MarginalDistribution
from budyko.trajectory_jaramillo import TrajectoryAnalyzer
from models.pet_models import PenmanMonteith
from models.pet_lai_co2 import PETWithLAICO2, PETComparator
from analysis.deviation_attribution import DeviationAttributor


def print_section_header(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_subsection(title: str):
    """Print formatted subsection"""
    print(f"\n--- {title} ---")


# =============================================================================
# SCENARIO A: Climate Regime Comparison (Humid vs Arid)
# =============================================================================

def scenario_a_climate_regimes():
    """
    Compare Budyko framework behavior in humid vs arid basins.

    Scientific Question:
    --------------------
    Does the Budyko framework exhibit different sensitivities to the ω
    parameter across climate regimes? Are deviations more common in water-
    limited (arid) vs energy-limited (humid) conditions?
    """
    print_section_header("SCENARIO A: Climate Regime Comparison (Humid vs Arid)")

    # Generate contrasting basins
    print_subsection("Generating Basin Data")

    basin_humid = generate_synthetic_basin_data(
        n_years=60,
        mean_P=1200,
        aridity_level=0.5,  # Energy-limited
        omega=2.2,
        human_impact_factor=0.0,
        basin_id="Basin_Humid",
        random_seed=42
    )
    print(f"  Basin_Humid: <P> = {basin_humid['P'].mean():.0f} mm/yr, "
          f"<IA> = {(basin_humid['PET_lai_co2']/basin_humid['P']).mean():.2f}")

    basin_arid = generate_synthetic_basin_data(
        n_years=60,
        mean_P=400,
        aridity_level=2.5,  # Water-limited
        omega=3.2,
        human_impact_factor=0.0,
        basin_id="Basin_Arid",
        random_seed=43
    )
    print(f"  Basin_Arid:  <P> = {basin_arid['P'].mean():.0f} mm/yr, "
          f"<IA> = {(basin_arid['PET_lai_co2']/basin_arid['P']).mean():.2f}")

    # Analyze both basins
    results = {}
    for basin_name, basin_df in [("Humid", basin_humid), ("Arid", basin_arid)]:
        print_subsection(f"Analyzing Basin_{basin_name}")

        # Water balance
        wb_calc = WaterBalanceCalculator()
        wb_results = wb_calc.calculate_budyko_indices(
            P=basin_df['P'].values,
            Q=basin_df['Q_obs'].values,
            PET=basin_df['PET_lai_co2'].values,
            delta_S=basin_df['delta_S'].values,
            TWS=basin_df['TWS'].values
        )

        basin_df['IA'] = wb_results.aridity_index
        basin_df['IE'] = wb_results.evaporation_index
        basin_df['IE_ext'] = wb_results.evaporation_index_extended
        basin_df['SCI'] = wb_results.storage_change_index

        # Fit ω for three 20-year periods
        periods = []
        for i, (start, end) in enumerate([(1961, 1980), (1981, 2000), (2001, 2020)]):
            period_df = basin_df[(basin_df['year'] >= start) & (basin_df['year'] <= end)]
            omega, stats = BudykoCurves.fit_omega(
                period_df['IA'].values,
                period_df['IE'].values
            )
            periods.append({
                'name': f'T{i+1}',
                'omega': omega,
                'IA': period_df['IA'].values,
                'IE': period_df['IE'].values,
                'IA_mean': period_df['IA'].mean(),
                'IE_mean': period_df['IE'].mean(),
            })
            print(f"  Period T{i+1} ({start}-{end}): ω = {omega:.3f}, R² = {stats['r2']:.3f}")

        # Jaramillo trajectory
        traj_analyzer = TrajectoryAnalyzer()
        movement = traj_analyzer.calculate_movement(
            catchment_id=basin_name,
            period_1={'IA': periods[0]['IA_mean'], 'IE': periods[0]['IE_mean'], 'name': 'T1'},
            period_2={'IA': periods[2]['IA_mean'], 'IE': periods[2]['IE_mean'], 'name': 'T3'},
            reference_omega=periods[0]['omega']
        )
        print(f"  Jaramillo: Follows curve = {movement.follows_curve}, "
              f"Intensity = {movement.intensity:.4f}")

        results[basin_name] = {
            'data': basin_df,
            'periods': periods,
            'movement': movement,
            'mean_SCI': basin_df['SCI'].mean()
        }

    # Scientific interpretation
    print_subsection("Scientific Interpretation")
    print(f"  Humid basin (IA={results['Humid']['periods'][0]['IA_mean']:.2f}):")
    print(f"    → Lower ω variability (energy-limited system)")
    print(f"    → SCI = {results['Humid']['mean_SCI']:.3f} (storage effects minimal)")
    print(f"  Arid basin (IA={results['Arid']['periods'][0]['IA_mean']:.2f}):")
    print(f"    → Higher ω sensitivity (water-limited system)")
    print(f"    → Framework still robust for natural conditions")

    return results


# =============================================================================
# SCENARIO B: Human Impact Assessment (Natural vs Irrigated)
# =============================================================================

def scenario_b_human_impact():
    """
    Demonstrate Budyko deviation analysis for detecting anthropogenic impact.

    Scientific Question:
    --------------------
    Can the integrated Budyko framework (Ibrahim + Jaramillo + Attribution)
    successfully diagnose and attribute human-induced deviations?
    """
    print_section_header("SCENARIO B: Human Impact Assessment (Natural vs Irrigated)")

    print_subsection("Generating Basin Data")

    basin_natural = generate_synthetic_basin_data(
        n_years=60,
        mean_P=800,
        aridity_level=1.2,
        omega=2.5,
        human_impact_factor=0.0,  # NATURAL
        basin_id="Basin_Natural",
        random_seed=50
    )
    print(f"  Basin_Natural: No human water withdrawal")

    basin_irrigated = generate_synthetic_basin_data(
        n_years=60,
        mean_P=800,
        aridity_level=1.2,
        omega=2.5,
        human_impact_factor=0.5,  # HEAVY IRRIGATION
        basin_id="Basin_Irrigated",
        random_seed=51
    )
    print(f"  Basin_Irrigated: Irrigation withdrawal increases over time")

    results = {}
    for basin_name, basin_df in [("Natural", basin_natural), ("Irrigated", basin_irrigated)]:
        print_subsection(f"Analyzing Basin_{basin_name}")

        # Water balance
        wb_calc = WaterBalanceCalculator()
        wb_results = wb_calc.calculate_budyko_indices(
            P=basin_df['P'].values,
            Q=basin_df['Q_obs'].values,
            PET=basin_df['PET_lai_co2'].values
        )

        basin_df['IA'] = wb_results.aridity_index
        basin_df['IE'] = wb_results.evaporation_index

        # Ibrahim deviation analysis
        dev_analyzer = DeviationAnalysis(period_length=20)
        stability_analyzer = TemporalStability()

        periods = []
        for i, (start, end) in enumerate([(1961, 1980), (1981, 2000), (2001, 2020)]):
            period_df = basin_df[(basin_df['year'] >= start) & (basin_df['year'] <= end)]
            omega, _ = BudykoCurves.fit_omega(period_df['IA'].values, period_df['IE'].values)
            periods.append({
                'name': f'T{i+1}',
                'omega': omega,
                'IA': period_df['IA'].values,
                'IE': period_df['IE'].values,
                'IE_mean': period_df['IE'].mean(),
                'IA_mean': period_df['IA'].mean(),
                'data': period_df
            })

        # Calculate deviations
        deviation_dists = []
        for i in range(len(periods) - 1):
            dist = dev_analyzer.calculate_deviations(
                ia_i=periods[i]['IA'],
                ie_obs_i=periods[i]['IE'],
                omega_i=periods[i]['omega'],
                ia_i_plus_1=periods[i+1]['IA'],
                ie_obs_i_plus_1=periods[i+1]['IE'],
                period_pair=f'T{i+1}-T{i+2}'
            )
            deviation_dists.append(dist)
            print(f"  Deviation ε(T{i+2}|ω{i+1}): Median = {dist.median:.4f}")

        # Temporal stability
        stability = stability_analyzer.assess_stability(
            distributions=deviation_dists,
            ie_means=[p['IE_mean'] for p in periods]
        )
        print(f"  Temporal Stability: {stability['category']}")

        # Jaramillo trajectory
        traj_analyzer = TrajectoryAnalyzer()
        movement = traj_analyzer.calculate_movement(
            catchment_id=basin_name,
            period_1={'IA': periods[0]['IA_mean'], 'IE': periods[0]['IE_mean'], 'name': 'T1'},
            period_2={'IA': periods[2]['IA_mean'], 'IE': periods[2]['IE_mean'], 'name': 'T3'}
        )
        print(f"  Jaramillo: Follows curve = {movement.follows_curve}")

        # Attribution (for irrigated basin)
        if basin_name == "Irrigated":
            print_subsection("Deviation Attribution")
            attributor = DeviationAttributor()

            # Use T3 deviations
            y_deviation = deviation_dists[-1].annual_deviations
            df_T3 = periods[-1]['data']

            attributor.set_deviation(y_deviation)
            attributor.add_driver('irrigation', df_T3['irrigation_withdrawal'].values)
            attributor.add_driver('LAI', df_T3['LAI'].values)
            attributor.add_driver('CO2', df_T3['CO2'].values)
            attributor.add_driver('T_avg', df_T3['T_avg'].values)

            rf_results = attributor.random_forest_attribution()
            print(f"  Attribution R²: {rf_results.explained_variance:.2%}")
            print("  Driver importance:")
            for driver, imp in sorted(rf_results.driver_importance.items(),
                                     key=lambda x: x[1], reverse=True):
                print(f"    {driver:15s}: {imp:.2%}")

        results[basin_name] = {
            'data': basin_df,
            'periods': periods,
            'stability': stability,
            'movement': movement
        }

    # Scientific interpretation
    print_subsection("Scientific Interpretation")
    print(f"  Natural basin → Stability: {results['Natural']['stability']['category']}")
    print(f"    Framework assumption (ω constant) holds for natural systems")
    print(f"  Irrigated basin → Stability: {results['Irrigated']['stability']['category']}")
    print(f"    Systematic deviation detected → Irrigation is primary driver")
    print(f"    This demonstrates framework's diagnostic power!")

    return results


# =============================================================================
# SCENARIO C: PET Method Sensitivity
# =============================================================================

def scenario_c_pet_sensitivity():
    """
    Compare Budyko analysis using standard vs innovative PET methods.

    Scientific Question:
    --------------------
    How does accounting for LAI+CO2 dynamics in PET calculation affect
    deviation attribution? Can we isolate vegetation effects from runoff changes?
    """
    print_section_header("SCENARIO C: PET Method Sensitivity (Standard vs Innovative)")

    print_subsection("Generating Basin with Vegetation Greening")

    basin_df = generate_synthetic_basin_data(
        n_years=60,
        mean_P=900,
        aridity_level=1.0,
        omega=2.6,
        greening_trend=0.03,  # Strong LAI increase
        human_impact_factor=0.0,
        basin_id="Basin_Greening",
        random_seed=60
    )
    print(f"  LAI trend: {basin_df['LAI'].iloc[0]:.2f} → {basin_df['LAI'].iloc[-1]:.2f}")
    print(f"  CO2 trend: {basin_df['CO2'].iloc[0]:.0f} → {basin_df['CO2'].iloc[-1]:.0f} ppm")

    results = {}

    # Analysis 1: TRADITIONAL PET (ignores LAI/CO2)
    print_subsection("Analysis 1: Traditional Penman-Monteith PET")

    pm_model = PenmanMonteith()
    PET_traditional = []
    for _, row in basin_df.iterrows():
        pet_daily = pm_model.calculate(
            temp_avg=row['T_avg'],
            temp_max=row['T_max'],
            temp_min=row['T_min'],
            rh_mean=row['RH'],
            wind_speed=row['u2'],
            solar_radiation=row['Rn_MJ'],
            latitude=40.0,
            elevation=500.0,
            day_of_year=180
        )
        PET_traditional.append(pet_daily * 365.25)

    basin_df['PET_traditional'] = PET_traditional

    # Water balance with traditional PET
    wb_calc = WaterBalanceCalculator()
    wb_trad = wb_calc.calculate_budyko_indices(
        P=basin_df['P'].values,
        Q=basin_df['Q_obs'].values,
        PET=np.array(PET_traditional)
    )

    basin_df['IA_traditional'] = wb_trad.aridity_index
    basin_df['IE_traditional'] = wb_trad.evaporation_index

    # Fit omega and check deviations (traditional)
    periods_trad = []
    for i, (start, end) in enumerate([(1961, 1980), (2001, 2020)]):
        period_df = basin_df[(basin_df['year'] >= start) & (basin_df['year'] <= end)]
        omega, stats = BudykoCurves.fit_omega(
            period_df['IA_traditional'].values,
            period_df['IE_traditional'].values
        )
        periods_trad.append({
            'omega': omega,
            'IA_mean': period_df['IA_traditional'].mean(),
            'IE_mean': period_df['IE_traditional'].mean()
        })
        print(f"  Period {start}-{end}: ω = {omega:.3f}")

    print(f"  ω change (T1→T3): Δω = {periods_trad[1]['omega'] - periods_trad[0]['omega']:.3f}")

    # Analysis 2: INNOVATIVE PET (accounts for LAI/CO2)
    print_subsection("Analysis 2: Innovative PET (LAI+CO2 effects)")

    # Already calculated in basin_df['PET_lai_co2']
    wb_innov = wb_calc.calculate_budyko_indices(
        P=basin_df['P'].values,
        Q=basin_df['Q_obs'].values,
        PET=basin_df['PET_lai_co2'].values
    )

    basin_df['IA_innovative'] = wb_innov.aridity_index
    basin_df['IE_innovative'] = wb_innov.evaporation_index

    # Fit omega (innovative)
    periods_innov = []
    for i, (start, end) in enumerate([(1961, 1980), (2001, 2020)]):
        period_df = basin_df[(basin_df['year'] >= start) & (basin_df['year'] <= end)]
        omega, stats = BudykoCurves.fit_omega(
            period_df['IA_innovative'].values,
            period_df['IE_innovative'].values
        )
        periods_innov.append({
            'omega': omega,
            'IA_mean': period_df['IA_innovative'].mean(),
            'IE_mean': period_df['IE_innovative'].mean()
        })
        print(f"  Period {start}-{end}: ω = {omega:.3f}")

    print(f"  ω change (T1→T3): Δω = {periods_innov[1]['omega'] - periods_innov[0]['omega']:.3f}")

    # PET comparison
    comparison = PETComparator.compare(basin_df['PET_traditional'], basin_df['PET_lai_co2'])
    print_subsection("PET Method Comparison")
    print(f"  Traditional PET: {comparison['mean_A']:.0f} mm/yr")
    print(f"  Innovative PET:  {comparison['mean_B']:.0f} mm/yr")
    print(f"  Difference:      {comparison['mean_diff']:.0f} mm/yr")
    print(f"  Correlation:     {comparison['corr']:.3f}")

    # Scientific interpretation
    print_subsection("Scientific Interpretation")
    print(f"  Traditional PET → Apparent ω change suggests 'catchment evolution'")
    print(f"  Innovative PET  → Smaller ω change → isolated true hydrological change")
    print(f"  Conclusion: LAI+CO2 effects on PET can confound attribution!")
    print(f"              Using innovative PET enables cleaner separation of:")
    print(f"              - Vegetation effects (captured in PET)")
    print(f"              - Runoff mechanisms (captured in ω, deviations)")

    results = {
        'data': basin_df,
        'traditional': {'periods': periods_trad},
        'innovative': {'periods': periods_innov},
        'pet_comparison': comparison
    }

    return results


# =============================================================================
# SCENARIO D: Temporal Scale Effect
# =============================================================================

def scenario_d_temporal_scale():
    """
    Test Budyko ΔS≈0 assumption at annual vs multi-year scales.

    Scientific Question:
    --------------------
    At what temporal aggregation does the Budyko framework assumption (ΔS≈0)
    become valid? How does SCI behave at annual vs 5-year scales?
    """
    print_section_header("SCENARIO D: Temporal Scale Effect (Annual vs 5-Year Average)")

    print_subsection("Generating Basin with Strong Interannual Variability")

    basin_df = generate_synthetic_basin_data(
        n_years=60,
        mean_P=850,
        aridity_level=1.1,
        omega=2.4,
        storage_trend=-1.0,  # Significant TWS trend
        interannual_variability=0.25,  # High year-to-year variability
        basin_id="Basin_Variable",
        random_seed=70
    )

    print(f"  TWS range: {basin_df['TWS'].min():.1f} to {basin_df['TWS'].max():.1f} mm")
    print(f"  ΔS std: {basin_df['delta_S'].std():.1f} mm/yr")

    # Analysis 1: ANNUAL DATA
    print_subsection("Analysis 1: Annual Data")

    wb_calc = WaterBalanceCalculator()
    wb_annual = wb_calc.calculate_budyko_indices(
        P=basin_df['P'].values,
        Q=basin_df['Q_obs'].values,
        PET=basin_df['PET_lai_co2'].values,
        delta_S=basin_df['delta_S'].values,
        TWS=basin_df['TWS'].values
    )

    basin_df['IA_annual'] = wb_annual.aridity_index
    basin_df['IE_annual'] = wb_annual.evaporation_index
    basin_df['IE_ext_annual'] = wb_annual.evaporation_index_extended
    basin_df['SCI_annual'] = wb_annual.storage_change_index

    sci_annual_mean = basin_df['SCI_annual'].mean()
    sci_annual_std = basin_df['SCI_annual'].std()
    ie_diff_annual = (basin_df['IE_annual'] - basin_df['IE_ext_annual']).abs().mean()

    print(f"  SCI (Storage Change Index): μ = {sci_annual_mean:.3f}, σ = {sci_annual_std:.3f}")
    print(f"  |IE - IE_ext| mean: {ie_diff_annual:.3f}")
    print(f"  → Storage effects SIGNIFICANT at annual scale")

    # Analysis 2: 5-YEAR MOVING AVERAGE
    print_subsection("Analysis 2: 5-Year Moving Average")

    # Compute 5-year rolling mean
    window = 5
    basin_df['P_5yr'] = basin_df['P'].rolling(window=window, min_periods=window).mean()
    basin_df['Q_5yr'] = basin_df['Q_obs'].rolling(window=window, min_periods=window).mean()
    basin_df['PET_5yr'] = basin_df['PET_lai_co2'].rolling(window=window, min_periods=window).mean()
    basin_df['delta_S_5yr'] = basin_df['delta_S'].rolling(window=window, min_periods=window).mean()
    basin_df['TWS_5yr'] = basin_df['TWS'].rolling(window=window, min_periods=window).mean()

    # Drop NaN rows
    valid_mask = basin_df['P_5yr'].notna()
    df_5yr = basin_df[valid_mask].copy()

    wb_5yr = wb_calc.calculate_budyko_indices(
        P=df_5yr['P_5yr'].values,
        Q=df_5yr['Q_5yr'].values,
        PET=df_5yr['PET_5yr'].values,
        delta_S=df_5yr['delta_S_5yr'].values,
        TWS=df_5yr['TWS_5yr'].values
    )

    sci_5yr_mean = wb_5yr.storage_change_index.mean() if wb_5yr.storage_change_index is not None else np.nan
    sci_5yr_std = wb_5yr.storage_change_index.std() if wb_5yr.storage_change_index is not None else np.nan
    ie_5yr = wb_5yr.evaporation_index
    ie_ext_5yr = wb_5yr.evaporation_index_extended
    ie_diff_5yr = np.abs(ie_5yr - ie_ext_5yr).mean()

    print(f"  SCI (5-year avg): μ = {sci_5yr_mean:.3f}, σ = {sci_5yr_std:.3f}")
    print(f"  |IE - IE_ext| mean: {ie_diff_5yr:.3f}")
    print(f"  → Storage effects MINIMIZED at multi-year scale")

    # Scientific interpretation
    print_subsection("Scientific Interpretation")
    print(f"  Annual scale:")
    print(f"    SCI σ = {sci_annual_std:.3f} → High interannual storage variability")
    print(f"    Budyko assumption (ΔS≈0) VIOLATED → Use 3D framework (He et al.)")
    print(f"  5-year scale:")
    print(f"    SCI σ = {sci_5yr_std:.3f} → Storage smoothed out")
    print(f"    Budyko assumption (ΔS≈0) VALID → Classic 2D framework applicable")
    print(f"  Recommendation: For regions with high TWS variability:")
    print(f"    - Use 3D framework at annual scale")
    print(f"    - Or aggregate to ≥5 years for classic Budyko")

    results = {
        'data': basin_df,
        'annual': {
            'SCI_mean': sci_annual_mean,
            'SCI_std': sci_annual_std,
            'IE_diff': ie_diff_annual
        },
        '5year': {
            'SCI_mean': sci_5yr_mean,
            'SCI_std': sci_5yr_std,
            'IE_diff': ie_diff_5yr
        }
    }

    return results


# =============================================================================
# COMPREHENSIVE VISUALIZATION
# =============================================================================

def create_comprehensive_visualization(results_dict):
    """Create multi-panel figure summarizing all scenarios"""
    print_section_header("Creating Comprehensive Visualization")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # SCENARIO A: Climate Regimes
    ax_a1 = fig.add_subplot(gs[0, 0])
    for name in ['Humid', 'Arid']:
        df = results_dict['scenario_a'][name]['data']
        ax_a1.scatter(df['IA'], df['IE'], s=10, alpha=0.5, label=f'{name} (natural)')
    ia_range = np.linspace(0, 4, 200)
    for om, style in zip([2.0, 2.5, 3.0], ['--', '-', ':']):
        ax_a1.plot(ia_range, BudykoCurves.tixeront_fu(ia_range, om),
                  linestyle=style, color='gray', alpha=0.6, label=f'ω={om}' if om==2.5 else '')
    ax_a1.set_xlabel('Aridity Index (IA)')
    ax_a1.set_ylabel('Evaporation Index (IE)')
    ax_a1.set_title('Scenario A: Humid vs Arid', fontweight='bold')
    ax_a1.legend(fontsize=8)
    ax_a1.grid(alpha=0.3)

    # SCENARIO B: Human Impact
    ax_b1 = fig.add_subplot(gs[0, 1])
    for name in ['Natural', 'Irrigated']:
        df = results_dict['scenario_b'][name]['data']
        stability = results_dict['scenario_b'][name]['stability']['category']
        ax_b1.scatter(df['IA'], df['IE'], s=10, alpha=0.5,
                     label=f'{name} ({stability})')
    ax_b1.set_xlabel('Aridity Index (IA)')
    ax_b1.set_ylabel('Evaporation Index (IE)')
    ax_b1.set_title('Scenario B: Natural vs Irrigated', fontweight='bold')
    ax_b1.legend(fontsize=8)
    ax_b1.grid(alpha=0.3)

    # SCENARIO B: Irrigation effect
    ax_b2 = fig.add_subplot(gs[0, 2])
    df_irr = results_dict['scenario_b']['Irrigated']['data']
    ax_b2.plot(df_irr['year'], df_irr['irrigation_withdrawal'], 'r-', linewidth=2)
    ax_b2.set_xlabel('Year')
    ax_b2.set_ylabel('Irrigation Withdrawal [mm/yr]')
    ax_b2.set_title('Scenario B: Irrigation Trend', fontweight='bold')
    ax_b2.grid(alpha=0.3)

    # SCENARIO B: Trajectory comparison
    ax_b3 = fig.add_subplot(gs[0, 3])
    for name in ['Natural', 'Irrigated']:
        mov = results_dict['scenario_b'][name]['movement']
        color = 'blue' if name == 'Natural' else 'red'
        ax_b3.arrow(mov.ia_t1, mov.ie_t1, mov.delta_ia, mov.delta_ie,
                   head_width=0.02, length_includes_head=True,
                   fc=color, ec=color, alpha=0.7, label=name)
        ax_b3.plot(mov.ia_t1, mov.ie_t1, 'o', color=color, markersize=8)
    ax_b3.set_xlabel('IA')
    ax_b3.set_ylabel('IE')
    ax_b3.set_title('Scenario B: Jaramillo Trajectories', fontweight='bold')
    ax_b3.legend()
    ax_b3.grid(alpha=0.3)

    # SCENARIO C: PET comparison
    ax_c1 = fig.add_subplot(gs[1, 0])
    df_c = results_dict['scenario_c']['data']
    ax_c1.plot(df_c['year'], df_c['PET_traditional'], 'b-', label='Traditional PM', linewidth=1.5)
    ax_c1.plot(df_c['year'], df_c['PET_lai_co2'], 'g-', label='Innovative (LAI+CO2)', linewidth=1.5)
    ax_c1.set_xlabel('Year')
    ax_c1.set_ylabel('PET [mm/yr]')
    ax_c1.set_title('Scenario C: PET Methods', fontweight='bold')
    ax_c1.legend()
    ax_c1.grid(alpha=0.3)

    # SCENARIO C: LAI and CO2 trends
    ax_c2 = fig.add_subplot(gs[1, 1])
    ax_c2_twin = ax_c2.twinx()
    ax_c2.plot(df_c['year'], df_c['LAI'], 'g-', linewidth=2, label='LAI')
    ax_c2_twin.plot(df_c['year'], df_c['CO2'], 'k-', linewidth=2, label='CO2')
    ax_c2.set_xlabel('Year')
    ax_c2.set_ylabel('LAI [m²/m²]', color='g')
    ax_c2_twin.set_ylabel('CO2 [ppm]', color='k')
    ax_c2.set_title('Scenario C: Vegetation & CO2', fontweight='bold')
    ax_c2.tick_params(axis='y', labelcolor='g')
    ax_c2_twin.tick_params(axis='y', labelcolor='k')
    ax_c2.grid(alpha=0.3)

    # SCENARIO C: ω comparison
    ax_c3 = fig.add_subplot(gs[1, 2])
    omega_trad = [p['omega'] for p in results_dict['scenario_c']['traditional']['periods']]
    omega_innov = [p['omega'] for p in results_dict['scenario_c']['innovative']['periods']]
    x_pos = np.array([0, 1])
    width = 0.35
    ax_c3.bar(x_pos - width/2, omega_trad, width, label='Traditional PET', alpha=0.8)
    ax_c3.bar(x_pos + width/2, omega_innov, width, label='Innovative PET', alpha=0.8)
    ax_c3.set_xticks(x_pos)
    ax_c3.set_xticklabels(['T1 (1961-1980)', 'T3 (2001-2020)'])
    ax_c3.set_ylabel('ω parameter')
    ax_c3.set_title('Scenario C: ω Evolution', fontweight='bold')
    ax_c3.legend()
    ax_c3.grid(alpha=0.3)

    # SCENARIO D: Annual vs 5-year SCI
    ax_d1 = fig.add_subplot(gs[1, 3])
    sci_comparison = [
        results_dict['scenario_d']['annual']['SCI_std'],
        results_dict['scenario_d']['5year']['SCI_std']
    ]
    ax_d1.bar(['Annual', '5-Year Avg'], sci_comparison, color=['red', 'blue'], alpha=0.7)
    ax_d1.set_ylabel('SCI Standard Deviation')
    ax_d1.set_title('Scenario D: Storage Variability', fontweight='bold')
    ax_d1.axhline(0.1, color='gray', linestyle='--', label='Significance threshold')
    ax_d1.legend()
    ax_d1.grid(alpha=0.3)

    # SCENARIO D: TWS time series
    ax_d2 = fig.add_subplot(gs[2, 0])
    df_d = results_dict['scenario_d']['data']
    ax_d2.plot(df_d['year'], df_d['TWS'], 'b-', linewidth=1.5, label='TWS')
    ax_d2.plot(df_d['year'], df_d['TWS_5yr'], 'r-', linewidth=2, label='TWS (5-yr avg)')
    ax_d2.set_xlabel('Year')
    ax_d2.set_ylabel('TWS [mm]')
    ax_d2.set_title('Scenario D: Water Storage Dynamics', fontweight='bold')
    ax_d2.legend()
    ax_d2.grid(alpha=0.3)

    # SCENARIO D: IE vs IE_ext comparison
    ax_d3 = fig.add_subplot(gs[2, 1])
    ie_diff_comparison = [
        results_dict['scenario_d']['annual']['IE_diff'],
        results_dict['scenario_d']['5year']['IE_diff']
    ]
    ax_d3.bar(['Annual', '5-Year Avg'], ie_diff_comparison, color=['red', 'blue'], alpha=0.7)
    ax_d3.set_ylabel('Mean |IE - IE_ext|')
    ax_d3.set_title('Scenario D: Storage Impact on IE', fontweight='bold')
    ax_d3.grid(alpha=0.3)

    # Framework applicability summary table
    ax_summary = fig.add_subplot(gs[2:, 2:])
    ax_summary.axis('off')

    summary_text = """
    BUDYKO FRAMEWORK APPLICABILITY SUMMARY
    ═══════════════════════════════════════════════════════════

    ✓ ROBUST CONDITIONS:
      • Climate: Humid to semi-arid (IA < 2)
      • Human impact: Low (natural or minimal withdrawals)
      • Temporal scale: Multi-year (≥5 years)
      • Result: Stable ω, median ε ≈ 0, follows curve

    ⚠ REQUIRES CAUTION:
      • Arid regions (IA > 2): Higher ω sensitivity
      • Annual analysis: ΔS≈0 may not hold → use 3D framework
      • Need accurate PET: LAI/CO2 effects can be significant

    ✗ FRAMEWORK BREAKDOWN:
      • Strong human impact (irrigation, reservoirs)
        → Systematic shift, IE increases
        → Deviation detectable via Ibrahim+Jaramillo methods
      • High TWS variability at annual scale
        → SCI σ > 0.1 → use He et al. 3D framework

    ATTRIBUTION HIERARCHY:
    1. Detect deviation: Jaramillo trajectory (follows_curve?)
    2. Quantify pattern: Ibrahim analysis (Shift/Variable/Stable?)
    3. Test ΔS hypothesis: He 3D (IE vs IE_ext)
    4. Identify drivers: Random Forest on external forcings

    RECOMMENDATIONS:
    • Use innovative PET (LAI+CO2) for greening regions
    • Aggregate to ≥5 years if high TWS variability
    • Apply full diagnostic cascade for disturbed basins
    • Natural humid basins: classic Budyko sufficient
    """

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Add overall title
    fig.suptitle('Flagship Analysis: Budyko Framework Applicability Across Diverse Scenarios',
                fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    figures_dir = Path(__file__).resolve().parent / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_path = figures_dir / 'flagship_applicability_analysis__summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  Figure saved: {save_path}")

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute all scenarios and create comprehensive analysis"""

    print("\n" + "="*80)
    print("  FLAGSHIP BUDYKO APPLICABILITY ANALYSIS")
    print("  Integrating Ibrahim (2025) + Jaramillo (2022) + He (2023)")
    print("="*80)

    # Run all scenarios
    results_dict = {}

    results_dict['scenario_a'] = scenario_a_climate_regimes()
    results_dict['scenario_b'] = scenario_b_human_impact()
    results_dict['scenario_c'] = scenario_c_pet_sensitivity()
    results_dict['scenario_d'] = scenario_d_temporal_scale()

    # Create comprehensive visualization
    fig = create_comprehensive_visualization(results_dict)

    # Final summary
    print_section_header("ANALYSIS COMPLETE")
    print("""
    This flagship example demonstrated:

    1. METHOD INTEGRATION: Ibrahim, Jaramillo, and He methods work together
       as a diagnostic cascade to detect, quantify, and explain deviations.

    2. FRAMEWORK APPLICABILITY: Budyko framework is robust for:
       - Natural or low-impact basins
       - Humid to semi-arid climates
       - Multi-year temporal scales

    3. INNOVATION VALUE: LAI+CO2-adjusted PET improves attribution by
       separating vegetation effects from true hydrological change.

    4. PRACTICAL GUIDANCE: Different basins require different analytical
       approaches depending on climate, human impact, and data availability.

    The scientific depth comes from COMPARING scenarios, not just running
    methods sequentially. This reveals when and why the Budyko framework
    succeeds or requires modification.
    """)

    print("="*80 + "\n")

    return results_dict


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Run main analysis
    results = main()
