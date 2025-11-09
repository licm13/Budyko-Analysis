#!/usr/bin/env python3
"""
Synthetic Basin Data Simulation Utilities

Provides unified, highly configurable functions for generating synthetic
hydrological data for Budyko analysis testing and demonstration.

This module eliminates code duplication across examples and provides
a consistent interface for creating diverse basin scenarios.
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BasinCharacteristics:
    """Configuration for synthetic basin simulation.

    Attributes
    ----------
    mean_precipitation : float
        Mean annual precipitation [mm/year]
    aridity_level : float
        Target aridity index (PET/P). Controls how water-limited the basin is.
        - < 0.5: Humid (energy-limited)
        - 0.5-1.0: Sub-humid
        - 1.0-2.0: Semi-arid
        - > 2.0: Arid (water-limited)
    omega : float
        Budyko catchment parameter (controls partitioning)
    warming_trend : float
        Temperature increase rate [°C/year]
    greening_trend : float
        LAI increase rate [m²/m²/year]
    human_impact_factor : float
        Human water withdrawal as fraction of natural runoff [0-1]
        0 = natural, 1 = complete withdrawal
    storage_trend : float
        TWS depletion/increase rate [mm/year]
    interannual_variability : float
        Coefficient of variation for climate variables
    """
    mean_precipitation: float = 800.0
    aridity_level: float = 1.2
    omega: float = 2.5
    warming_trend: float = 0.03
    greening_trend: float = 0.02
    human_impact_factor: float = 0.0
    storage_trend: float = -0.5
    interannual_variability: float = 0.15


class SyntheticBasinSimulator:
    """Generate synthetic basin data for Budyko analysis.

    This class provides methods to create realistic synthetic hydrological
    time series that exhibit various trends and characteristics useful for
    testing Budyko framework applicability.

    Examples
    --------
    >>> # Create a natural humid basin
    >>> sim = SyntheticBasinSimulator()
    >>> chars = BasinCharacteristics(
    ...     mean_precipitation=1200,
    ...     aridity_level=0.6,
    ...     human_impact_factor=0.0
    ... )
    >>> df = sim.generate_basin_data(n_years=60, characteristics=chars)

    >>> # Create an arid basin with irrigation
    >>> chars_arid = BasinCharacteristics(
    ...     mean_precipitation=400,
    ...     aridity_level=2.5,
    ...     human_impact_factor=0.4
    ... )
    >>> df_arid = sim.generate_basin_data(n_years=60, characteristics=chars_arid)
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize simulator.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_basin_data(
        self,
        n_years: int = 60,
        start_year: int = 1961,
        characteristics: Optional[BasinCharacteristics] = None,
        basin_id: str = "synthetic_basin"
    ) -> pd.DataFrame:
        """Generate comprehensive synthetic basin data.

        This is the main entry point for data generation. It creates a complete
        time series including:
        - Climate forcing (P, T, RH, wind, radiation)
        - Vegetation dynamics (LAI)
        - Atmospheric composition (CO2)
        - Water storage (TWS, ΔS)
        - Runoff (natural and observed with human impact)
        - PET (with LAI+CO2 effects)

        Parameters
        ----------
        n_years : int
            Length of simulation [years]
        start_year : int
            Starting year of simulation
        characteristics : BasinCharacteristics, optional
            Basin configuration. Uses defaults if None.
        basin_id : str
            Basin identifier

        Returns
        -------
        pd.DataFrame
            Synthetic data with columns:
            - year: Year
            - P: Precipitation [mm/year]
            - Q_nat: Natural runoff [mm/year]
            - Q_obs: Observed runoff (with human impact) [mm/year]
            - T_avg, T_max, T_min: Temperature [°C]
            - RH: Relative humidity [%]
            - u2: Wind speed [m/s]
            - Rn_MJ, Rn_W: Net radiation [MJ/m²/day, W/m²]
            - LAI: Leaf area index [m²/m²]
            - CO2: Atmospheric CO2 [ppm]
            - TWS: Terrestrial water storage anomaly [mm]
            - delta_S: Storage change [mm/year]
            - irrigation_withdrawal: Human water withdrawal [mm/year]
            - PET_lai_co2: PET with LAI+CO2 effects [mm/year]
            - basin_id: Basin identifier
        """
        if characteristics is None:
            characteristics = BasinCharacteristics()

        years = np.arange(start_year, start_year + n_years)
        time_idx = np.arange(n_years)

        # 1. Climate forcing variables
        climate_data = self._generate_climate_forcing(
            n_years=n_years,
            time_idx=time_idx,
            mean_P=characteristics.mean_precipitation,
            warming_trend=characteristics.warming_trend,
            variability=characteristics.interannual_variability
        )

        # 2. Vegetation and atmospheric drivers
        vegetation_data = self._generate_vegetation_drivers(
            n_years=n_years,
            time_idx=time_idx,
            greening_trend=characteristics.greening_trend
        )

        # 3. Water storage
        storage_data = self._generate_storage_dynamics(
            n_years=n_years,
            time_idx=time_idx,
            storage_trend=characteristics.storage_trend
        )

        # 4. PET calculation (with LAI+CO2 effects)
        pet_data = self._calculate_pet_with_innovations(
            climate_data=climate_data,
            vegetation_data=vegetation_data
        )

        # 5. Runoff simulation (natural and observed)
        runoff_data = self._simulate_runoff(
            n_years=n_years,
            time_idx=time_idx,
            P=climate_data['P'],
            PET=pet_data['PET_lai_co2'],
            omega=characteristics.omega,
            aridity_level=characteristics.aridity_level,
            delta_S=storage_data['delta_S'],
            human_impact_factor=characteristics.human_impact_factor
        )

        # 6. Combine all data
        data = {
            'year': years,
            'basin_id': basin_id,
            **climate_data,
            **vegetation_data,
            **storage_data,
            **pet_data,
            **runoff_data
        }

        return pd.DataFrame(data)

    def _generate_climate_forcing(
        self,
        n_years: int,
        time_idx: np.ndarray,
        mean_P: float,
        warming_trend: float,
        variability: float
    ) -> Dict[str, np.ndarray]:
        """Generate climate forcing variables.

        Returns
        -------
        dict
            Climate variables: P, T_avg, T_max, T_min, RH, u2, Rn_MJ, Rn_W
        """
        # Precipitation: multidecadal oscillation + interannual variability
        P_oscillation = 0.2 * mean_P * np.sin(2 * np.pi * time_idx / 30)
        P_noise = variability * mean_P * np.random.randn(n_years)
        P = mean_P + P_oscillation + P_noise
        P = np.maximum(P, 0.3 * mean_P)  # Physical lower bound

        # Temperature: warming trend + seasonal-like variation + noise
        T_base = 15.0
        T_warming = warming_trend * time_idx
        T_variation = 0.5 * np.random.randn(n_years)
        T_avg = T_base + T_warming + T_variation
        T_max = T_avg + 6.0
        T_min = T_avg - 6.0

        # Relative humidity: slight drying trend in warming climate
        RH_base = 65.0
        RH_trend = -0.1 * time_idx  # Drying
        RH_noise = 5.0 * np.random.randn(n_years)
        RH = RH_base + RH_trend + RH_noise
        RH = np.clip(RH, 40.0, 95.0)

        # Wind speed: stationary with small variability
        u2 = 2.0 + 0.3 * np.random.randn(n_years)
        u2 = np.maximum(u2, 0.5)

        # Net radiation: multidecadal variation (solar cycles)
        Rn_base = 7.0  # MJ/m²/day
        Rn_cycle = 0.5 * np.sin(2 * np.pi * time_idx / 20)
        Rn_noise = 0.5 * np.random.randn(n_years)
        Rn_MJ = Rn_base + Rn_cycle + Rn_noise
        Rn_W = Rn_MJ / 0.0864  # Convert to W/m²

        return {
            'P': P,
            'T_avg': T_avg,
            'T_max': T_max,
            'T_min': T_min,
            'RH': RH,
            'u2': u2,
            'Rn_MJ': Rn_MJ,
            'Rn_W': Rn_W
        }

    def _generate_vegetation_drivers(
        self,
        n_years: int,
        time_idx: np.ndarray,
        greening_trend: float
    ) -> Dict[str, np.ndarray]:
        """Generate vegetation and atmospheric composition data.

        Returns
        -------
        dict
            Vegetation variables: LAI, CO2
        """
        # LAI: greening trend + seasonal variation + noise
        LAI_base = 2.5
        LAI_trend = greening_trend * time_idx
        LAI_noise = 0.3 * np.random.randn(n_years)
        LAI = LAI_base + LAI_trend + LAI_noise
        LAI = np.clip(LAI, 0.5, 6.0)

        # CO2: historical-like increase (~1.5-2 ppm/year)
        CO2_base = 320.0  # ~1960s level
        CO2_trend = 1.5 * time_idx
        CO2_noise = 2.0 * np.random.randn(n_years)
        CO2 = CO2_base + CO2_trend + CO2_noise
        CO2 = np.maximum(CO2, 300.0)

        return {
            'LAI': LAI,
            'CO2': CO2
        }

    def _generate_storage_dynamics(
        self,
        n_years: int,
        time_idx: np.ndarray,
        storage_trend: float
    ) -> Dict[str, np.ndarray]:
        """Generate terrestrial water storage dynamics.

        Returns
        -------
        dict
            Storage variables: TWS, delta_S
        """
        # TWS: multidecadal oscillation + long-term trend
        TWS_oscillation = 50.0 * np.sin(2 * np.pi * time_idx / 15)
        TWS_trend = storage_trend * time_idx
        TWS = TWS_oscillation + TWS_trend

        # Storage change (year-to-year)
        delta_S = np.diff(TWS, prepend=TWS[0])

        return {
            'TWS': TWS,
            'delta_S': delta_S
        }

    def _calculate_pet_with_innovations(
        self,
        climate_data: Dict[str, np.ndarray],
        vegetation_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Calculate PET with LAI and CO2 effects.

        Uses the innovative PETWithLAICO2 model.

        Returns
        -------
        dict
            PET variables: PET_lai_co2
        """
        # Import here to avoid circular dependency
        from models.pet_lai_co2 import PETWithLAICO2

        pet_calc = PETWithLAICO2(elevation=500.0, latitude=40.0)

        PET_daily = pet_calc.calculate(
            temperature=climate_data['T_avg'],
            humidity=climate_data['RH'],
            wind_speed=climate_data['u2'],
            radiation=climate_data['Rn_W'],
            lai=vegetation_data['LAI'],
            co2=vegetation_data['CO2']
        )

        # Convert daily to annual
        PET_annual = PET_daily * 365.25

        return {
            'PET_lai_co2': PET_annual
        }

    def _simulate_runoff(
        self,
        n_years: int,
        time_idx: np.ndarray,
        P: np.ndarray,
        PET: np.ndarray,
        omega: float,
        aridity_level: float,
        delta_S: np.ndarray,
        human_impact_factor: float
    ) -> Dict[str, np.ndarray]:
        """Simulate natural and observed runoff.

        Natural runoff follows Budyko curve.
        Observed runoff includes human water withdrawal.

        Returns
        -------
        dict
            Runoff variables: Q_nat, Q_obs, irrigation_withdrawal
        """
        # Import here to avoid circular dependency
        from budyko.curves import BudykoCurves

        # Calculate natural runoff using Budyko framework
        # First adjust PET to match target aridity level
        mean_IA_current = np.mean(PET / P)
        PET_adjusted = PET * (aridity_level / mean_IA_current)

        # Natural evaporation following Budyko curve
        IA_nat = PET_adjusted / P
        IE_nat = BudykoCurves.tixeront_fu(IA_nat, omega)
        EA_nat = IE_nat * P
        Q_nat = P - EA_nat
        Q_nat = np.maximum(Q_nat, 0.0)

        # Human water withdrawal (increases over time)
        if human_impact_factor > 0:
            # Gradual increase in water use
            withdrawal_trend = human_impact_factor * (time_idx / n_years)
            withdrawal_noise = 0.1 * human_impact_factor * np.random.randn(n_years)
            irrigation_fraction = withdrawal_trend + withdrawal_noise
            irrigation_fraction = np.clip(irrigation_fraction, 0, human_impact_factor)

            irrigation_withdrawal = irrigation_fraction * Q_nat
        else:
            irrigation_withdrawal = np.zeros(n_years)

        # Observed runoff = Natural - Withdrawal
        # Also accounting for storage change in water balance
        Q_obs = P - EA_nat - delta_S - irrigation_withdrawal
        Q_obs = np.maximum(Q_obs, 0.05 * Q_nat)  # Minimum 5% of natural runoff

        return {
            'Q_nat': Q_nat,
            'Q_obs': Q_obs,
            'irrigation_withdrawal': irrigation_withdrawal
        }


def generate_synthetic_basin_data(
    n_years: int = 60,
    start_year: int = 1961,
    mean_P: float = 800.0,
    aridity_level: float = 1.2,
    omega: float = 2.5,
    warming_trend: float = 0.03,
    greening_trend: float = 0.02,
    human_impact_factor: float = 0.0,
    storage_trend: float = -0.5,
    interannual_variability: float = 0.15,
    basin_id: str = "synthetic_basin",
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """Convenience function for generating synthetic basin data.

    This is a simplified interface to SyntheticBasinSimulator for common use cases.

    Parameters
    ----------
    n_years : int
        Number of years to simulate
    start_year : int
        Starting year
    mean_P : float
        Mean annual precipitation [mm/year]
    aridity_level : float
        Target aridity index (PET/P)
    omega : float
        Budyko catchment parameter
    warming_trend : float
        Temperature increase rate [°C/year]
    greening_trend : float
        LAI increase rate [m²/m²/year]
    human_impact_factor : float
        Fraction of water withdrawn [0-1]
    storage_trend : float
        TWS change rate [mm/year]
    interannual_variability : float
        Coefficient of variation for climate
    basin_id : str
        Basin identifier
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Synthetic basin data

    Examples
    --------
    >>> # Humid basin, no human impact
    >>> df_humid = generate_synthetic_basin_data(
    ...     mean_P=1200, aridity_level=0.6, human_impact_factor=0.0
    ... )

    >>> # Arid basin with heavy irrigation
    >>> df_arid = generate_synthetic_basin_data(
    ...     mean_P=400, aridity_level=2.5, human_impact_factor=0.5
    ... )
    """
    characteristics = BasinCharacteristics(
        mean_precipitation=mean_P,
        aridity_level=aridity_level,
        omega=omega,
        warming_trend=warming_trend,
        greening_trend=greening_trend,
        human_impact_factor=human_impact_factor,
        storage_trend=storage_trend,
        interannual_variability=interannual_variability
    )

    simulator = SyntheticBasinSimulator(random_seed=random_seed)
    return simulator.generate_basin_data(
        n_years=n_years,
        start_year=start_year,
        characteristics=characteristics,
        basin_id=basin_id
    )


__all__ = [
    'BasinCharacteristics',
    'SyntheticBasinSimulator',
    'generate_synthetic_basin_data'
]
