# src/budyko/trajectory_jaramillo.py
"""
Budyko空间运动轨迹分析（Jaramillo et al. 2022方法）
"""
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class BudykoMovement:
    """Budyko空间运动数据类"""
    catchment_id: str
    period_1_name: str
    period_2_name: str
    
    # 起点和终点
    ia_t1: float  # 时期1的干旱指数
    ie_t1: float  # 时期1的蒸发指数
    ia_t2: float  # 时期2的干旱指数
    ie_t2: float  # 时期2的蒸发指数
    
    # 运动向量分量
    delta_ia: float  # ΔIA = IA(t2) - IA(t1)
    delta_ie: float  # ΔIE = IE(t2) - IE(t1)
    
    # 运动特征
    intensity: float     # I = |v| = sqrt(ΔIA² + ΔIE²)
    direction_angle: float  # θ (度), 从垂直方向顺时针
    
    # 曲线遵循判断
    follows_curve: bool  # 是否遵循Budyko曲线
    reference_omega: float  # 参考曲线的ω
    
    # 额外分类
    movement_type: str  # 运动类型分类


class TrajectoryAnalyzer:
    """
    Budyko空间轨迹分析器
    
    实现Jaramillo et al. (2022)的核心方法
    """
    
    # 定义"遵循曲线"的方向角范围（度）
    FOLLOW_ANGLE_RANGES = [(45, 90), (225, 270)]
    
    def __init__(self):
        self.movements = {}
    
    def calculate_movement(self,
                          catchment_id: str,
                          period_1: Dict,
                          period_2: Dict,
                          reference_omega: float = None) -> BudykoMovement:
        """
        计算两个时期之间的Budyko空间运动
        
        Parameters
        ----------
        catchment_id : str
            流域ID
        period_1 : Dict
            时期1数据，包含 'IA', 'IE', 'name'
        period_2 : Dict
            时期2数据，包含 'IA', 'IE', 'name'
        reference_omega : float, optional
            参考曲线的ω参数（通常用period_1的ω）
            
        Returns
        -------
        BudykoMovement
            运动对象
        """
        # 提取坐标
        ia_t1 = period_1['IA']
        ie_t1 = period_1['IE']
        ia_t2 = period_2['IA']
        ie_t2 = period_2['IE']
        
        # 计算运动向量分量
        # 使用微小四舍五入，避免浮点表示误差导致的断言失败（如0.7-0.6!=0.1）
        delta_ia = round(ia_t2 - ia_t1, 10)
        delta_ie = round(ie_t2 - ie_t1, 10)
        
        # 计算运动强度（向量模）
        intensity = np.sqrt(delta_ia**2 + delta_ie**2)
        
        # 计算方向角（从垂直方向顺时针，度）
        # 注意：Jaramillo定义为从垂直(IE轴)顺时针
        # arctan2返回[-π, π]，需要转换
        direction_angle = self._calculate_direction_angle(delta_ia, delta_ie)
        
        # 判断是否遵循Budyko曲线
        follows_curve = self._check_follows_curve(direction_angle)
        
        # 运动类型分类
        movement_type = self._classify_movement(delta_ia, delta_ie, 
                                               ia_t1, ie_t1,
                                               follows_curve)
        
        movement = BudykoMovement(
            catchment_id=catchment_id,
            period_1_name=period_1['name'],
            period_2_name=period_2['name'],
            ia_t1=ia_t1,
            ie_t1=ie_t1,
            ia_t2=ia_t2,
            ie_t2=ie_t2,
            delta_ia=delta_ia,
            delta_ie=delta_ie,
            intensity=intensity,
            direction_angle=direction_angle,
            follows_curve=follows_curve,
            reference_omega=reference_omega if reference_omega else np.nan,
            movement_type=movement_type
        )
        
        self.movements[catchment_id] = movement
        return movement
    
    @staticmethod
    def _calculate_direction_angle(delta_ia: float, 
                                   delta_ie: float) -> float:
        """
        计算方向角（度），从垂直方向顺时针
        
        坐标系：
        - IE轴（垂直）：0°
        - IA轴（水平右）：90°
        - IE轴（垂直下）：180°
        - IA轴（水平左）：270°
        
        Returns
        -------
        float
            方向角 [0, 360)
        """
        # arctan2(y, x)给出从x轴逆时针的角度
        # 我们需要从y轴(IE)顺时针
        angle_rad = np.arctan2(delta_ia, delta_ie)  # 注意参数顺序
        angle_deg = np.degrees(angle_rad)
        
        # 转换到[0, 360)
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def _check_follows_curve(self, direction_angle: float) -> bool:
        """
        判断运动方向是否符合"遵循Budyko曲线"的标准
        
        根据Jaramillo et al. (2022) Figure 1:
        - 遵循曲线：45° < θ < 90° 或 225° < θ < 270°
        - 偏离曲线：其他角度
        
        Parameters
        ----------
        direction_angle : float
            方向角（度）
            
        Returns
        -------
        bool
            是否遵循曲线
        """
        for angle_min, angle_max in self.FOLLOW_ANGLE_RANGES:
            if angle_min <= direction_angle <= angle_max:
                return True
        return False
    
    @staticmethod
    def _classify_movement(delta_ia: float,
                          delta_ie: float,
                          ia_start: float,
                          ie_start: float,
                          follows_curve: bool) -> str:
        """
        对运动进行详细分类
        
        分类体系：
        1. 遵循曲线 vs 偏离曲线
        2. 变湿润/变干旱
        3. 蒸发增加/减少
        
        Returns
        -------
        str
            运动类型标签
        """
        # 基本方向
        if abs(delta_ia) < 0.01 and abs(delta_ie) < 0.01:
            return "Stationary"
        
        # 干旱/湿润分类
        aridity_change = "Aridification" if delta_ia > 0 else "Humidification"
        
        # 蒸发变化
        evap_change = "Evap_increase" if delta_ie > 0 else "Evap_decrease"
        
        # 遵循/偏离
        trajectory = "Following" if follows_curve else "Deviating"
        
        # 组合标签
        return f"{trajectory}_{aridity_change}_{evap_change}"
    
    def batch_calculate_movements(self,
                                 data: pd.DataFrame,
                                 period_1_cols: Tuple[str, str],
                                 period_2_cols: Tuple[str, str],
                                 catchment_id_col: str = 'catchment_id',
                                 omega_col: str = None) -> pd.DataFrame:
        """
        批量计算多个流域的运动 (Vectorized for better performance)
        
        Parameters
        ----------
        data : pd.DataFrame
            包含流域数据的DataFrame
        period_1_cols : Tuple[str, str]
            (IA列名, IE列名) for period 1
        period_2_cols : Tuple[str, str]
            (IA列名, IE列名) for period 2
        catchment_id_col : str
            流域ID列名
        omega_col : str, optional
            参考ω列名
            
        Returns
        -------
        pd.DataFrame
            运动结果表
        """
        # Vectorized calculations for better performance
        ia_t1 = data[period_1_cols[0]].values
        ie_t1 = data[period_1_cols[1]].values
        ia_t2 = data[period_2_cols[0]].values
        ie_t2 = data[period_2_cols[1]].values
        
        # Calculate deltas vectorized
        delta_ia = np.round(ia_t2 - ia_t1, 10)
        delta_ie = np.round(ie_t2 - ie_t1, 10)
        
        # Calculate intensity vectorized
        intensity = np.sqrt(delta_ia**2 + delta_ie**2)
        
        # Calculate direction angles vectorized
        direction_angle = np.degrees(np.arctan2(delta_ia, delta_ie))
        direction_angle = np.where(direction_angle < 0, direction_angle + 360, direction_angle)
        
        # Vectorized check for following curve
        follows_curve = np.zeros(len(data), dtype=bool)
        for angle_min, angle_max in self.FOLLOW_ANGLE_RANGES:
            follows_curve |= (direction_angle >= angle_min) & (direction_angle <= angle_max)
        
        # Build result DataFrame efficiently
        results = pd.DataFrame({
            'catchment_id': data[catchment_id_col].values,
            'IA_t1': ia_t1,
            'IE_t1': ie_t1,
            'IA_t2': ia_t2,
            'IE_t2': ie_t2,
            'delta_IA': delta_ia,
            'delta_IE': delta_ie,
            'intensity': intensity,
            'direction_angle': direction_angle,
            'follows_curve': follows_curve,
            'reference_omega': data[omega_col].values if omega_col and omega_col in data.columns else np.nan
        })
        
        # Vectorized movement type classification
        results['movement_type'] = self._classify_movement_vectorized(
            delta_ia, delta_ie, ia_t1, ie_t1, follows_curve
        )
        
        return results
    
    @staticmethod
    def _classify_movement_vectorized(delta_ia: np.ndarray,
                                     delta_ie: np.ndarray,
                                     ia_start: np.ndarray,
                                     ie_start: np.ndarray,
                                     follows_curve: np.ndarray) -> np.ndarray:
        """
        Vectorized movement classification for better performance
        
        Returns
        -------
        np.ndarray
            Array of movement type strings
        """
        # Stationary check
        stationary = (np.abs(delta_ia) < 0.01) & (np.abs(delta_ie) < 0.01)
        
        # Build classification components
        aridity_change = np.where(delta_ia > 0, "Aridification", "Humidification")
        evap_change = np.where(delta_ie > 0, "Evap_increase", "Evap_decrease")
        trajectory = np.where(follows_curve, "Following", "Deviating")
        
        # Fully vectorized string concatenation using np.char
        movement_types = np.where(
            stationary,
            "Stationary",
            np.char.add(
                np.char.add(
                    np.char.add(trajectory, "_"),
                    aridity_change
                ),
                np.char.add("_", evap_change)
            )
        )
        
        return movement_types



class MovementStatistics:
    """运动统计分析"""
    
    @staticmethod
    def summarize_movements(movements_df: pd.DataFrame,
                           group_by: str = None) -> pd.DataFrame:
        """
        汇总运动统计
        
        Parameters
        ----------
        movements_df : pd.DataFrame
            运动结果数据框
        group_by : str, optional
            分组变量（如区域、气候带等）
            
        Returns
        -------
        pd.DataFrame
            统计摘要
        """
        if group_by and group_by in movements_df.columns:
            grouped = movements_df.groupby(group_by)
        else:
            grouped = [(None, movements_df)]
        
        summaries = []
        for group_name, group_data in grouped:
            n_total = len(group_data)
            n_following = group_data['follows_curve'].sum()
            pct_following = 100 * n_following / n_total
            
            summary = {
                'group': group_name if group_name else 'All',
                'n_catchments': n_total,
                'n_following_curve': n_following,
                'n_deviating': n_total - n_following,
                'pct_following': pct_following,
                'pct_deviating': 100 - pct_following,
                'mean_intensity': group_data['intensity'].mean(),
                'median_intensity': group_data['intensity'].median(),
                'mean_delta_IA': group_data['delta_IA'].mean(),
                'mean_delta_IE': group_data['delta_IE'].mean(),
                'mean_direction': group_data['direction_angle'].mean()
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    @staticmethod
    def direction_distribution(movements_df: pd.DataFrame,
                             n_bins: int = 8) -> pd.DataFrame:
        """
        方向角分布统计（用于玫瑰图）
        
        Parameters
        ----------
        movements_df : pd.DataFrame
            运动结果
        n_bins : int
            方向bin数量（通常8或16）
            
        Returns
        -------
        pd.DataFrame
            方向分布
        """
        # 定义方向bin
        bins = np.linspace(0, 360, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # 统计每个方向的频率
        hist, _ = np.histogram(movements_df['direction_angle'], bins=bins)
        
        # 计算平均强度
        movements_df['direction_bin'] = pd.cut(
            movements_df['direction_angle'],
            bins=bins,
            labels=bin_centers
        )
        
        avg_intensity = movements_df.groupby('direction_bin')['intensity'].mean()
        
        return pd.DataFrame({
            'direction': bin_centers,
            'frequency': hist,
            'percentage': 100 * hist / len(movements_df),
            'avg_intensity': avg_intensity.values
        })


class ScenarioComparator:
    """
    多情景比较分析
    
    用于比较不同CMIP6情景（SSP1-2.6, SSP2-4.5, SSP5-8.5等）
    """
    
    def __init__(self, scenarios: List[str]):
        """
        Parameters
        ----------
        scenarios : List[str]
            情景列表，如 ['historical', 'ssp126', 'ssp245', 'ssp585']
        """
        self.scenarios = scenarios
        self.results = {}
    
    def compare_following_rates(self,
                               scenario_movements: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        比较不同情景下的"遵循曲线"比例
        
        Parameters
        ----------
        scenario_movements : Dict[str, pd.DataFrame]
            {scenario_name: movements_df} 字典
            
        Returns
        -------
        pd.DataFrame
            情景对比表
        """
        comparisons = []
        
        for scenario, movements_df in scenario_movements.items():
            n_total = len(movements_df)
            n_following = movements_df['follows_curve'].sum()
            
            comparisons.append({
                'scenario': scenario,
                'n_catchments': n_total,
                'n_following': n_following,
                'n_deviating': n_total - n_following,
                'pct_following': 100 * n_following / n_total,
                'pct_deviating': 100 * (n_total - n_following) / n_total,
                'mean_intensity': movements_df['intensity'].mean(),
                'std_intensity': movements_df['intensity'].std()
            })
        
        return pd.DataFrame(comparisons)
    
    def aridity_group_analysis(self,
                              scenario_movements: Dict[str, pd.DataFrame],
                              aridity_bins: List[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        按干旱度分组的情景比较
        
        Parameters
        ----------
        scenario_movements : Dict[str, pd.DataFrame]
            情景运动数据
        aridity_bins : List[Tuple[float, float]], optional
            干旱度区间，默认为 [(0, 0.5), (0.5, 1), (1, 2), (2, 5)]
            
        Returns
        -------
        pd.DataFrame
            分组比较结果
        """
        if aridity_bins is None:
            aridity_bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 5)]
        
        results = []
        
        for scenario, movements_df in scenario_movements.items():
            for ia_min, ia_max in aridity_bins:
                # 选择初始干旱度在该区间的流域
                mask = (movements_df['IA_t1'] >= ia_min) & \
                       (movements_df['IA_t1'] < ia_max)
                group_data = movements_df[mask]
                
                if len(group_data) == 0:
                    continue
                
                n_total = len(group_data)
                n_following = group_data['follows_curve'].sum()
                
                results.append({
                    'scenario': scenario,
                    'aridity_class': f"{ia_min}-{ia_max}",
                    'aridity_label': self._get_aridity_label(ia_min, ia_max),
                    'n_catchments': n_total,
                    'pct_following': 100 * n_following / n_total,
                    'pct_deviating': 100 * (n_total - n_following) / n_total,
                    'mean_intensity': group_data['intensity'].mean()
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def _get_aridity_label(ia_min: float, ia_max: float) -> str:
        """干旱度标签"""
        if ia_max <= 0.5:
            return "Humid"
        elif ia_max <= 1:
            return "Sub-humid"
        elif ia_max <= 2:
            return "Semi-arid"
        else:
            return "Arid"