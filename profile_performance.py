#!/usr/bin/env python
"""
Performance profiling script to identify bottlenecks in Budyko-Analysis
"""
import numpy as np
import pandas as pd
import time
from typing import Callable
import sys

def time_function(func: Callable, name: str, *args, **kwargs):
    """Time a function and print results"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start
    print(f"  {name}: {duration:.4f}s")
    return result, duration

def profile_budyko_curves():
    """Profile Budyko curve calculations"""
    print("\n=== Budyko Curve Calculations ===")
    from src.budyko.curves import BudykoCurves
    
    # Small dataset
    ia_small = np.random.uniform(0.5, 3.0, 100)
    omega = 2.6
    _, t1 = time_function(
        lambda: [BudykoCurves.tixeront_fu(ia_small, omega) for _ in range(1000)],
        "tixeront_fu (100 points, 1000 iters)"
    )
    
    # Large dataset
    ia_large = np.random.uniform(0.5, 3.0, 10000)
    _, t2 = time_function(
        lambda: [BudykoCurves.tixeront_fu(ia_large, omega) for _ in range(100)],
        "tixeront_fu (10k points, 100 iters)"
    )
    
    # Omega fitting
    ia_vals = np.random.uniform(0.5, 3.0, 20)
    ie_vals = np.random.uniform(0.3, 0.7, 20)
    _, t3 = time_function(
        lambda: [BudykoCurves.fit_omega(ia_vals, ie_vals) for _ in range(100)],
        "fit_omega (20 points, 100 iters)"
    )
    
    return {'tixeront_small': t1, 'tixeront_large': t2, 'fit_omega': t3}

def profile_pet_calculations():
    """Profile PET calculation methods"""
    print("\n=== PET Calculations ===")
    from src.models.pet_lai_co2 import PETWithLAICO2
    from src.models.pet_models import PenmanMonteithPET
    
    # Generate test data
    n = 365
    temperature = np.random.uniform(10, 25, n)
    humidity = np.random.uniform(40, 80, n)
    wind_speed = np.random.uniform(1, 4, n)
    radiation = np.random.uniform(100, 300, n)
    lai = np.random.uniform(2, 5, n)
    co2 = np.full(n, 400.0)
    
    # PETWithLAICO2
    pet_lai_co2 = PETWithLAICO2(elevation=100, latitude=35)
    _, t1 = time_function(
        lambda: [pet_lai_co2.calculate(temperature, humidity, wind_speed, 
                                       radiation, lai, co2) for _ in range(100)],
        "PETWithLAICO2 (365 days, 100 iters)"
    )
    
    # Standard Penman-Monteith
    temp_max = temperature + 5
    temp_min = temperature - 5
    solar_rad = radiation * 0.0864  # W/m2 to MJ/m2/day
    day_of_year = np.arange(1, n+1)
    
    pet_pm = PenmanMonteithPET()
    _, t2 = time_function(
        lambda: [pet_pm.calculate(temperature, temp_max, temp_min, humidity,
                                  wind_speed, solar_rad, 35, 100, day_of_year)
                for _ in range(100)],
        "PenmanMonteithPET (365 days, 100 iters)"
    )
    
    return {'pet_lai_co2': t1, 'pet_pm': t2}

def profile_deviation_analysis():
    """Profile deviation analysis"""
    print("\n=== Deviation Analysis ===")
    from src.budyko.deviation import DeviationAnalysis, TemporalStability
    
    # Generate test data
    n_years = 20
    ia_1 = np.random.uniform(0.5, 2.0, n_years)
    ie_1 = np.random.uniform(0.3, 0.7, n_years)
    ia_2 = np.random.uniform(0.5, 2.0, n_years)
    ie_2 = np.random.uniform(0.3, 0.7, n_years)
    omega = 2.6
    
    analyzer = DeviationAnalysis(period_length=20)
    
    _, t1 = time_function(
        lambda: [analyzer.calculate_deviations(ia_1, ie_1, omega, ia_2, ie_2, 'test')
                for _ in range(100)],
        "calculate_deviations (20 years, 100 iters)"
    )
    
    # Wilcoxon test
    dist = analyzer.calculate_deviations(ia_1, ie_1, omega, ia_2, ie_2, 'test')
    _, t2 = time_function(
        lambda: [analyzer.wilcoxon_test(dist) for _ in range(100)],
        "wilcoxon_test (100 iters)"
    )
    
    return {'calc_deviations': t1, 'wilcoxon': t2}

def profile_trajectory_analysis():
    """Profile trajectory analysis"""
    print("\n=== Trajectory Analysis ===")
    from src.budyko.trajectory_jaramillo import TrajectoryAnalyzer
    
    analyzer = TrajectoryAnalyzer()
    
    # Single movement calculation
    period_1 = {'IA': 0.8, 'IE': 0.5, 'name': 'P1'}
    period_2 = {'IA': 1.2, 'IE': 0.6, 'name': 'P2'}
    
    _, t1 = time_function(
        lambda: [analyzer.calculate_movement('basin_001', period_1, period_2)
                for _ in range(10000)],
        "calculate_movement (10k iters)"
    )
    
    # Batch calculation
    n_basins = 1000
    data = pd.DataFrame({
        'catchment_id': [f'basin_{i:03d}' for i in range(n_basins)],
        'IA_1': np.random.uniform(0.5, 2.0, n_basins),
        'IE_1': np.random.uniform(0.3, 0.7, n_basins),
        'IA_2': np.random.uniform(0.5, 2.0, n_basins),
        'IE_2': np.random.uniform(0.3, 0.7, n_basins),
    })
    
    _, t2 = time_function(
        lambda: analyzer.batch_calculate_movements(data, ('IA_1', 'IE_1'), ('IA_2', 'IE_2')),
        "batch_calculate_movements (1000 basins)"
    )
    
    return {'single_movement': t1, 'batch_movements': t2}

def profile_parallel_processing():
    """Profile parallel processing"""
    print("\n=== Parallel Processing ===")
    from src.utils.parallel_processing import ParallelBudykoAnalyzer
    
    def dummy_analysis(catchment_id: str, **kwargs) -> dict:
        # Simulate some work
        _ = np.random.random(100).sum()
        return {'catchment_id': catchment_id, 'value': np.random.random()}
    
    sizes = [100, 500, 1000]
    process_counts = [1, 2, 4]
    
    results = {}
    for size in sizes:
        for n_proc in process_counts:
            ids = [f'basin_{i:05d}' for i in range(size)]
            analyzer = ParallelBudykoAnalyzer(n_processes=n_proc, verbose=False)
            
            start = time.perf_counter()
            analyzer.process_catchments(ids, dummy_analysis)
            duration = time.perf_counter() - start
            
            key = f"size_{size}_proc_{n_proc}"
            results[key] = duration
            print(f"  {size} catchments, {n_proc} processes: {duration:.4f}s")
    
    return results

def profile_data_operations():
    """Profile data operations that might be slow"""
    print("\n=== Data Operations ===")
    
    # DataFrame operations
    n_rows = 100000
    df = pd.DataFrame({
        'basin_id': np.repeat(np.arange(1000), 100),
        'date': pd.date_range('2000-01-01', periods=n_rows, freq='D'),
        'Q': np.random.uniform(0, 100, n_rows),
        'P': np.random.uniform(0, 200, n_rows),
        'T': np.random.uniform(-10, 35, n_rows),
    })
    
    # Groupby operations
    _, t1 = time_function(
        lambda: df.groupby('basin_id').agg({'Q': 'mean', 'P': 'sum', 'T': 'mean'}),
        "groupby + agg (100k rows, 1000 basins)"
    )
    
    # Filtering
    _, t2 = time_function(
        lambda: df[df['Q'] > 50],
        "filtering (100k rows)"
    )
    
    # Merging
    df2 = pd.DataFrame({
        'basin_id': np.arange(1000),
        'area': np.random.uniform(100, 5000, 1000),
    })
    _, t3 = time_function(
        lambda: df.merge(df2, on='basin_id'),
        "merge (100k x 1k rows)"
    )
    
    return {'groupby': t1, 'filter': t2, 'merge': t3}

def identify_bottlenecks(results: dict):
    """Identify the slowest operations"""
    print("\n=== Performance Summary ===")
    
    all_times = []
    for category, timings in results.items():
        if isinstance(timings, dict):
            for name, time_val in timings.items():
                all_times.append((f"{category}.{name}", time_val))
        else:
            all_times.append((category, timings))
    
    # Sort by time
    all_times.sort(key=lambda x: x[1], reverse=True)
    
    print("\nSlowest operations (top 10):")
    for i, (name, time_val) in enumerate(all_times[:10], 1):
        print(f"  {i}. {name}: {time_val:.4f}s")

def main():
    print("=" * 60)
    print("Budyko-Analysis Performance Profiling")
    print("=" * 60)
    
    results = {}
    
    try:
        results['budyko_curves'] = profile_budyko_curves()
    except Exception as e:
        print(f"Error in budyko_curves: {e}")
    
    try:
        results['pet_calculations'] = profile_pet_calculations()
    except Exception as e:
        print(f"Error in pet_calculations: {e}")
    
    try:
        results['deviation_analysis'] = profile_deviation_analysis()
    except Exception as e:
        print(f"Error in deviation_analysis: {e}")
    
    try:
        results['trajectory_analysis'] = profile_trajectory_analysis()
    except Exception as e:
        print(f"Error in trajectory_analysis: {e}")
    
    try:
        results['parallel_processing'] = profile_parallel_processing()
    except Exception as e:
        print(f"Error in parallel_processing: {e}")
    
    try:
        results['data_operations'] = profile_data_operations()
    except Exception as e:
        print(f"Error in data_operations: {e}")
    
    identify_bottlenecks(results)

if __name__ == "__main__":
    main()
