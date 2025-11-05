"""
Test suite to verify performance optimizations maintain correctness
"""
import numpy as np
import pandas as pd
import pytest
from src.budyko.deviation import DeviationAnalysis
from src.budyko.curves import BudykoCurves
from src.budyko.trajectory_jaramillo import TrajectoryAnalyzer
from scipy import stats


class TestDeviationOptimizations:
    """Test that deviation analysis optimizations are correct"""
    
    def test_fast_skew_matches_scipy(self):
        """Verify our fast skewness calculation matches scipy"""
        np.random.seed(42)
        data = np.random.randn(20)
        
        analyzer = DeviationAnalysis()
        fast_skew = analyzer._fast_skew(data)
        scipy_skew = stats.skew(data)
        
        # Should be very close (within reasonable tolerance due to numerical differences)
        assert np.abs(fast_skew - scipy_skew) < 1e-6
    
    def test_fast_skew_edge_cases(self):
        """Test edge cases for fast skewness"""
        analyzer = DeviationAnalysis()
        
        # Small sample
        data_small = np.array([1.0, 2.0])
        assert analyzer._fast_skew(data_small) == 0.0
        
        # Zero variance
        data_constant = np.array([5.0, 5.0, 5.0, 5.0])
        assert analyzer._fast_skew(data_constant) == 0.0
    
    def test_calculate_deviations_correctness(self):
        """Ensure optimized calculate_deviations produces correct results"""
        np.random.seed(42)
        n_years = 20
        ia_1 = np.random.uniform(0.5, 2.0, n_years)
        ie_1 = np.random.uniform(0.3, 0.7, n_years)
        ia_2 = np.random.uniform(0.5, 2.0, n_years)
        ie_2 = np.random.uniform(0.3, 0.7, n_years)
        omega = 2.6
        
        analyzer = DeviationAnalysis(period_length=20)
        dist = analyzer.calculate_deviations(ia_1, ie_1, omega, ia_2, ie_2, 'test')
        
        # Check that all fields are computed
        assert dist.period_name == 'test'
        assert len(dist.annual_deviations) == n_years
        assert isinstance(dist.median, float)
        assert isinstance(dist.mean, float)
        assert isinstance(dist.std, float)
        assert isinstance(dist.iqr, float)
        assert isinstance(dist.skew, float)
        assert isinstance(dist.fitted_params, dict)
        
        # Verify statistical properties
        assert dist.iqr >= 0
        assert dist.std >= 0
        assert np.isclose(dist.median, np.median(dist.annual_deviations))
        assert np.isclose(dist.mean, np.mean(dist.annual_deviations))


class TestCurvesOptimizations:
    """Test that Budyko curve optimizations are correct"""
    
    def test_fit_omega_smart_guess(self):
        """Test that smart initial guesses still produce correct results"""
        np.random.seed(42)
        
        # Test different aridity regimes
        test_cases = [
            (np.random.uniform(0.3, 0.8, 20), 'humid'),  # Low aridity
            (np.random.uniform(0.8, 1.5, 20), 'semi-arid'),  # Medium aridity
            (np.random.uniform(1.5, 3.0, 20), 'arid'),  # High aridity
        ]
        
        for ia_vals, regime in test_cases:
            ie_vals = 1 + ia_vals - (1 + ia_vals**2.5)**(1/2.5) + np.random.randn(20) * 0.02
            omega_opt, result = BudykoCurves.fit_omega(ia_vals, ie_vals)
            
            # Check that omega is in valid range
            assert 0.1 <= omega_opt <= 10.0
            # Check that we get a reasonable fit
            assert result['r2'] > 0.8
            assert result['rmse'] < 0.2


class TestTrajectoryOptimizations:
    """Test that trajectory analysis optimizations are correct"""
    
    def test_batch_calculate_movements_vectorized(self):
        """Verify vectorized batch calculation produces same results"""
        np.random.seed(42)
        n_basins = 100
        
        data = pd.DataFrame({
            'catchment_id': [f'basin_{i:03d}' for i in range(n_basins)],
            'IA_1': np.random.uniform(0.5, 2.0, n_basins),
            'IE_1': np.random.uniform(0.3, 0.7, n_basins),
            'IA_2': np.random.uniform(0.5, 2.0, n_basins),
            'IE_2': np.random.uniform(0.3, 0.7, n_basins),
        })
        
        analyzer = TrajectoryAnalyzer()
        result = analyzer.batch_calculate_movements(data, ('IA_1', 'IE_1'), ('IA_2', 'IE_2'))
        
        # Check output shape and columns
        assert result.shape[0] == n_basins
        expected_cols = {'catchment_id', 'IA_t1', 'IE_t1', 'IA_t2', 'IE_t2',
                        'delta_IA', 'delta_IE', 'intensity', 'direction_angle',
                        'follows_curve', 'movement_type', 'reference_omega'}
        assert set(result.columns) == expected_cols
        
        # Verify a few calculations manually
        for i in range(min(5, n_basins)):
            row = result.iloc[i]
            expected_delta_ia = round(row['IA_t2'] - row['IA_t1'], 10)
            expected_delta_ie = round(row['IE_t2'] - row['IE_t1'], 10)
            
            assert np.isclose(row['delta_IA'], expected_delta_ia, atol=1e-9)
            assert np.isclose(row['delta_IE'], expected_delta_ie, atol=1e-9)
            
            expected_intensity = np.sqrt(expected_delta_ia**2 + expected_delta_ie**2)
            assert np.isclose(row['intensity'], expected_intensity, atol=1e-9)
    
    def test_classify_movement_vectorized_correctness(self):
        """Test vectorized movement classification"""
        n = 10
        delta_ia = np.array([0.1, -0.1, 0.2, -0.2, 0.005, 0.3, -0.3, 0.15, -0.15, 0.0])
        delta_ie = np.array([0.05, -0.05, 0.15, -0.15, 0.005, 0.2, -0.2, 0.1, -0.1, 0.0])
        ia_start = np.ones(n) * 1.0
        ie_start = np.ones(n) * 0.5
        follows_curve = np.array([True, False, True, False, False, True, False, True, False, False])
        
        analyzer = TrajectoryAnalyzer()
        movement_types = analyzer._classify_movement_vectorized(
            delta_ia, delta_ie, ia_start, ie_start, follows_curve
        )
        
        # Check we get string results
        assert len(movement_types) == n
        assert all(isinstance(mt, str) for mt in movement_types)
        
        # Check stationary case
        assert movement_types[4] == "Stationary" or movement_types[9] == "Stationary"
        
        # Check classification structure
        for mt in movement_types:
            if mt != "Stationary":
                assert "_" in mt
                parts = mt.split("_")
                assert parts[0] in ["Following", "Deviating"]
                assert "Aridification" in mt or "Humidification" in mt
                assert "Evap" in mt


class TestDataProcessingOptimizations:
    """Test that data processing optimizations are correct"""
    
    @pytest.mark.skipif(True, reason="Requires xarray which is optional dependency")
    def test_quality_control_runoff_vectorized(self):
        """Test that vectorized outlier detection works correctly"""
        from src.data_processing.basin_processor import BasinDataProcessor
        
        # Create test data with known outliers
        np.random.seed(42)
        n_basins = 10
        n_years = 20
        
        data_list = []
        for basin_id in range(n_basins):
            dates = pd.date_range('2000-01-01', periods=n_years, freq='Y')
            Q = np.random.uniform(50, 150, n_years)
            # Add outlier to first basin
            if basin_id == 0:
                Q[5] = 1000  # Extreme outlier
            
            basin_data = pd.DataFrame({
                'date': dates,
                'basin_id': f'basin_{basin_id}',
                'Q': Q
            })
            data_list.append(basin_data)
        
        data = pd.concat(data_list, ignore_index=True)
        
        processor = BasinDataProcessor()
        result = processor._quality_control_runoff(data)
        
        # Check that outlier was flagged
        assert 'Q_flag' in result.columns
        outliers = result[result['Q_flag'] == 'outlier']
        assert len(outliers) > 0
        
        # Check that negative values are handled
        data_with_neg = data.copy()
        data_with_neg.loc[0, 'Q'] = -10
        result_neg = processor._quality_control_runoff(data_with_neg)
        assert result_neg.loc[0, 'Q_flag'] == 'negative'
        assert pd.isna(result_neg.loc[0, 'Q'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
