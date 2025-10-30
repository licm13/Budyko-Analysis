# src/utils/parallel_processing.py
"""
高性能并行处理框架（支持上万流域）
"""
import multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable, List, Dict, Any
import logging

class ParallelBudykoAnalyzer:
    """
    并行Budyko分析器
    
    针对中国10000+小流域优化
    """
    
    def __init__(self, 
                 n_processes: int = None,
                 chunk_size: int = 100,
                 verbose: bool = True):
        """
        Parameters
        ----------
        n_processes : int, optional
            进程数，默认为CPU核心数-1
        chunk_size : int
            每个进程处理的流域数
        verbose : bool
            是否显示进度条
        """
        if n_processes is None:
            n_processes = max(1, mp.cpu_count() - 1)
        
        self.n_processes = n_processes
        self.chunk_size = chunk_size
        self.verbose = verbose
        
        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_catchments(self,
                          catchment_ids: List[str],
                          analysis_function: Callable,
                          data_loader: Callable = None,
                          **kwargs) -> pd.DataFrame:
        """
        并行处理多个流域
        
        Parameters
        ----------
        catchment_ids : List[str]
            流域ID列表
        analysis_function : Callable
            分析函数，签名: f(catchment_id, data, **kwargs) -> Dict
        data_loader : Callable, optional
            数据加载函数，签名: f(catchment_id) -> data
            如果为None，则假设数据已在kwargs中
        **kwargs
            传递给analysis_function的额外参数
            
        Returns
        -------
        pd.DataFrame
            所有流域的分析结果
        """
        n_catchments = len(catchment_ids)
        self.logger.info(f"Processing {n_catchments} catchments with {self.n_processes} processes")
        
        # 准备任务
        if data_loader is not None:
            # 需要为每个流域加载数据
            tasks = [(cid, data_loader, analysis_function, kwargs) 
                    for cid in catchment_ids]
            worker_func = self._worker_with_loader
        else:
            # 数据已准备好
            tasks = [(cid, analysis_function, kwargs) 
                    for cid in catchment_ids]
            worker_func = self._worker_no_loader
        
        # 并行处理
        with Pool(processes=self.n_processes) as pool:
            if self.verbose:
                results = list(tqdm(
                    pool.imap(worker_func, tasks, chunksize=self.chunk_size),
                    total=n_catchments,
                    desc="Analyzing catchments"
                ))
            else:
                results = pool.map(worker_func, tasks, chunksize=self.chunk_size)
        
        # 过滤失败的结果
        successful_results = [r for r in results if r is not None]
        failed_count = n_catchments - len(successful_results)
        
        if failed_count > 0:
            self.logger.warning(f"{failed_count} catchments failed to process")
        
        # 转换为DataFrame
        if successful_results:
            return pd.DataFrame(successful_results)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def _worker_with_loader(args):
        """带数据加载的worker"""
        catchment_id, data_loader, analysis_function, kwargs = args
        
        try:
            # 加载数据
            data = data_loader(catchment_id)
            
            # 分析
            result = analysis_function(catchment_id, data, **kwargs)
            
            # 确保返回字典包含catchment_id
            if isinstance(result, dict):
                result['catchment_id'] = catchment_id
                return result
            else:
                return None
        
        except Exception as e:
            logging.error(f"Error processing {catchment_id}: {e}")
            return None
    
    @staticmethod
    def _worker_no_loader(args):
        """无需加载数据的worker"""
        catchment_id, analysis_function, kwargs = args
        
        try:
            result = analysis_function(catchment_id, **kwargs)
            
            if isinstance(result, dict):
                result['catchment_id'] = catchment_id
                return result
            else:
                return None
        
        except Exception as e:
            logging.error(f"Error processing {catchment_id}: {e}")
            return None
    
    def batch_trajectory_analysis(self,
                                 catchments_df: pd.DataFrame,
                                 period_1_cols: Tuple[str, str],
                                 period_2_cols: Tuple[str, str]) -> pd.DataFrame:
        """
        批量轨迹分析的便捷方法
        
        Parameters
        ----------
        catchments_df : pd.DataFrame
            包含所有流域数据的DataFrame
        period_1_cols : Tuple[str, str]
            时期1的(IA, IE)列名
        period_2_cols : Tuple[str, str]
            时期2的(IA, IE)列名
            
        Returns
        -------
        pd.DataFrame
            轨迹分析结果
        """
        from ..budyko.trajectory_jaramillo import TrajectoryAnalyzer
        
        def analyze_one_catchment(catchment_id, **kwargs):
            """单个流域分析"""
            row = kwargs['data'].loc[kwargs['data']['catchment_id'] == catchment_id].iloc[0]
            
            analyzer = TrajectoryAnalyzer()
            
            period_1 = {
                'IA': row[kwargs['period_1_cols'][0]],
                'IE': row[kwargs['period_1_cols'][1]],
                'name': 'Period_1'
            }
            
            period_2 = {
                'IA': row[kwargs['period_2_cols'][0]],
                'IE': row[kwargs['period_2_cols'][1]],
                'name': 'Period_2'
            }
            
            movement = analyzer.calculate_movement(
                catchment_id=catchment_id,
                period_1=period_1,
                period_2=period_2
            )
            
            return {
                'intensity': movement.intensity,
                'direction_angle': movement.direction_angle,
                'follows_curve': movement.follows_curve,
                'movement_type': movement.movement_type,
                'delta_IA': movement.delta_ia,
                'delta_IE': movement.delta_ie
            }
        
        # 执行并行分析
        results = self.process_catchments(
            catchment_ids=catchments_df['catchment_id'].tolist(),
            analysis_function=analyze_one_catchment,
            data=catchments_df,
            period_1_cols=period_1_cols,
            period_2_cols=period_2_cols
        )
        
        return results


# 使用示例
def demo_parallel_10k_catchments():
    """演示：处理10000个流域"""
    
    # 1. 生成模拟数据（实际使用时替换为真实数据）
    n_catchments = 10000
    np.random.seed(42)
    
    catchments_data = pd.DataFrame({
        'catchment_id': [f"CH_{i:05d}" for i in range(n_catchments)],
        'IA_1950': np.random.uniform(0.3, 3.0, n_catchments),
        'IE_1950': np.random.uniform(0.2, 0.95, n_catchments),
        'IA_2020': np.random.uniform(0.3, 3.0, n_catchments),
        'IE_2020': np.random.uniform(0.2, 0.95, n_catchments),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_catchments)
    })
    
    # 2. 初始化并行分析器
    analyzer = ParallelBudykoAnalyzer(n_processes=8, chunk_size=200, verbose=True)
    
    # 3. 批量轨迹分析
    print("Starting parallel trajectory analysis for 10,000 catchments...")
    
    results = analyzer.batch_trajectory_analysis(
        catchments_df=catchments_data,
        period_1_cols=('IA_1950', 'IE_1950'),
        period_2_cols=('IA_2020', 'IE_2020')
    )
    
    # 4. 汇总统计
    print("\nResults Summary:")
    print(f"Total catchments processed: {len(results)}")
    print(f"Following curve: {results['follows_curve'].sum()} ({100*results['follows_curve'].mean():.1f}%)")
    print(f"Mean intensity: {results['intensity'].mean():.4f}")
    print(f"\nMovement type distribution:")
    print(results['movement_type'].value_counts())
    
    return results