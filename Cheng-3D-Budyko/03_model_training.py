"""
机器学习模型训练模块 (Machine Learning Model Training Module)
使用BRT模型训练α和Qb,p参数的区域化关系
Train BRT models to regionalize α and Qb,p parameters

Author: [Your Name]
Date: 2025-01-01
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 导入自定义工具函数 / Import custom utility functions
from utils import calculate_r2, calculate_rmse


# =============================================================================
# 配置参数 (Configuration Parameters)
# =============================================================================

class ModelConfig:
    """模型配置类 / Model configuration class"""
    
    # 数据路径 / Data paths
    PROCESSED_DATA_DIR = Path('../data/processed')
    MODEL_OUTPUT_DIR = Path('../results/models')
    
    # BRT超参数 (根据论文) / BRT hyperparameters (from paper)
    # tc (tree complexity) = 12
    # lr (learning rate) = 0.01
    # bf (bag fraction) = 0.50
    BRT_PARAMS = {
        'max_depth': 12,  # 对应tree complexity
        'learning_rate': 0.01,
        'subsample': 0.50,  # 对应bag fraction
        'colsample_bytree': 0.8,
        'n_estimators': 500,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 交叉验证参数 / Cross-validation parameters
    N_FOLDS = 10  # 10折交叉验证 / 10-fold cross-validation
    N_REPEATS = 10  # 重复10次 / Repeat 10 times
    
    # 特征列表 (13个流域属性) / Feature list (13 catchment properties)
    FEATURES = [
        # 气候 / Climate
        'TC',      # Air temperature / 气温
        'SAI',     # Seasonality and asynchrony index / 季节性指数
        'SWE',     # Snow water equivalent / 雪水当量
        
        # 植被 / Vegetation
        'NDVI',    # Normalized difference vegetation index / 归一化植被指数
        'WUE',     # Water use efficiency / 水分利用效率
        'LAI',     # Leaf area index / 叶面积指数
        'RD',      # Maximum rooting depth / 最大根深
        
        # 地形 / Topography
        'CTI',     # Topographic index / 地形指数
        'ELEV',    # Mean elevation / 平均海拔
        'SLO',     # Slope / 坡度
        
        # 土壤 / Soil
        'STHI',    # Average soil and sedimentary deposit thickness / 土壤厚度
        'SPO',     # Soil porosity / 土壤孔隙度
        
        # 人类活动 / Human activities
        'HFP'      # Human influence index / 人类影响指数
    ]


# =============================================================================
# 数据加载函数 (Data Loading Functions)
# =============================================================================

def load_training_data(data_dir: Path) -> pd.DataFrame:
    """
    加载训练数据
    Load training data
    
    Parameters
    ----------
    data_dir : Path
        数据目录路径 / Data directory path
        
    Returns
    -------
    df : pd.DataFrame
        包含所有特征和目标变量的数据框
        DataFrame containing all features and target variables
        
    Notes
    -----
    预期的数据列 / Expected data columns:
    - catchment_id: 流域ID / Catchment ID
    - alpha: Budyko参数 (从方程3估计) / Budyko parameter (estimated from Eq. 3)
    - Qbp_mm_yr: 潜在基流 (从方程4估计) / Potential baseflow (estimated from Eq. 4)
    - 13个特征列 / 13 feature columns (see FEATURES list)
    """
    print("正在加载训练数据... / Loading training data...")
    
    # 加载参数数据 / Load parameter data
    param_file = data_dir / 'catchment_parameters.csv'
    if not param_file.exists():
        raise FileNotFoundError(f"未找到参数文件 / Parameter file not found: {param_file}")
    
    df_params = pd.read_csv(param_file)
    print(f"  加载了 {len(df_params)} 个流域的参数数据")
    print(f"  Loaded parameter data for {len(df_params)} catchments")
    
    # 加载流域属性数据 / Load catchment properties data
    props_file = data_dir / 'catchment_properties.csv'
    if not props_file.exists():
        raise FileNotFoundError(f"未找到属性文件 / Properties file not found: {props_file}")
    
    df_props = pd.read_csv(props_file)
    print(f"  加载了 {len(df_props)} 个流域的属性数据")
    print(f"  Loaded property data for {len(df_props)} catchments")
    
    # 合并数据 / Merge data
    df = pd.merge(df_params, df_props, on='catchment_id', how='inner')
    print(f"  合并后共 {len(df)} 个流域")
    print(f"  {len(df)} catchments after merging")
    
    # 检查缺失值 / Check missing values
    missing_summary = df[ModelConfig.FEATURES + ['alpha', 'Qbp_mm_yr']].isnull().sum()
    if missing_summary.sum() > 0:
        print("\n警告: 发现缺失值 / Warning: Missing values found:")
        print(missing_summary[missing_summary > 0])
        
        # 移除含缺失值的行 / Remove rows with missing values
        df = df.dropna(subset=ModelConfig.FEATURES + ['alpha', 'Qbp_mm_yr'])
        print(f"移除缺失值后剩余 {len(df)} 个流域")
        print(f"  {len(df)} catchments remaining after removing missing values")
    
    return df


# =============================================================================
# BRT模型训练函数 (BRT Model Training Functions)
# =============================================================================

def train_brt_model(X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   params: Dict) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    训练单个BRT模型
    Train a single BRT model
    
    Parameters
    ----------
    X_train : np.ndarray
        训练特征 / Training features
    y_train : np.ndarray
        训练目标 / Training targets
    X_test : np.ndarray
        测试特征 / Testing features
    y_test : np.ndarray
        测试目标 / Testing targets
    params : dict
        模型超参数 / Model hyperparameters
        
    Returns
    -------
    model : xgboost.XGBRegressor
        训练好的模型 / Trained model
    metrics : dict
        性能指标 / Performance metrics
    """
    # 创建模型 / Create model
    model = xgb.XGBRegressor(**params)
    
    # 训练模型 / Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # 预测 / Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算性能指标 / Calculate performance metrics
    metrics = {
        'train_r2': calculate_r2(y_train, y_train_pred),
        'train_rmse': calculate_rmse(y_train, y_train_pred),
        'test_r2': calculate_r2(y_test, y_test_pred),
        'test_rmse': calculate_rmse(y_test, y_test_pred)
    }
    
    return model, metrics


def train_cv_models(df: pd.DataFrame,
                   target: str,
                   config: ModelConfig) -> Tuple[List, pd.DataFrame]:
    """
    使用交叉验证训练多个模型
    Train multiple models using cross-validation
    
    Parameters
    ----------
    df : pd.DataFrame
        训练数据 / Training data
    target : str
        目标变量名称 ('alpha' 或 'Qbp_mm_yr')
        Target variable name ('alpha' or 'Qbp_mm_yr')
    config : ModelConfig
        配置对象 / Configuration object
        
    Returns
    -------
    models : list
        训练好的模型列表 / List of trained models
    metrics_df : pd.DataFrame
        性能指标汇总 / Performance metrics summary
    """
    print(f"\n训练 {target} 的BRT模型...")
    print(f"Training BRT models for {target}...")
    
    # 准备数据 / Prepare data
    X = df[config.FEATURES].values
    y = df[target].values
    
    # 初始化存储 / Initialize storage
    models = []
    all_metrics = []
    
    # K折交叉验证 / K-fold cross-validation
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    
    fold_num = 0
    for train_idx, test_idx in kf.split(X):
        fold_num += 1
        
        # 分割数据 / Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练模型 / Train model
        model, metrics = train_brt_model(
            X_train, y_train,
            X_test, y_test,
            config.BRT_PARAMS
        )
        
        # 存储 / Store
        models.append(model)
        metrics['fold'] = fold_num
        metrics['target'] = target
        all_metrics.append(metrics)
        
        print(f"  折 {fold_num}/{config.N_FOLDS}: " +
              f"训练 R²={metrics['train_r2']:.3f}, " +
              f"测试 R²={metrics['test_r2']:.3f}, " +
              f"RMSE={metrics['test_rmse']:.3f}")
        print(f"  Fold {fold_num}/{config.N_FOLDS}: " +
              f"Train R²={metrics['train_r2']:.3f}, " +
              f"Test R²={metrics['test_r2']:.3f}, " +
              f"RMSE={metrics['test_rmse']:.3f}")
    
    # 汇总性能 / Summarize performance
    metrics_df = pd.DataFrame(all_metrics)
    print(f"\n{target} 模型平均性能 / Average performance:")
    print(f"  测试 R²: {metrics_df['test_r2'].mean():.3f} ± {metrics_df['test_r2'].std():.3f}")
    print(f"  测试 RMSE: {metrics_df['test_rmse'].mean():.3f} ± {metrics_df['test_rmse'].std():.3f}")
    
    return models, metrics_df


# =============================================================================
# 主函数 (Main Function)
# =============================================================================

def main():
    """主训练流程 / Main training workflow"""
    
    print("="*80)
    print("开始机器学习模型训练 / Starting Machine Learning Model Training")
    print("="*80)
    
    config = ModelConfig()
    
    # 创建输出目录 / Create output directory
    config.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 步骤1: 加载训练数据 / Step 1: Load training data
    print("\n[步骤1/4] 加载训练数据...")
    print("[Step 1/4] Loading training data...")
    
    try:
        df = load_training_data(config.PROCESSED_DATA_DIR)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 01_data_preprocessing.py 和 02_parameter_estimation.py")
        print("Please run 01_data_preprocessing.py and 02_parameter_estimation.py first")
        return
    
    # 步骤2: 训练α模型 / Step 2: Train α models
    print("\n[步骤2/4] 训练α参数的BRT模型...")
    print("[Step 2/4] Training BRT models for α parameter...")
    
    alpha_models, alpha_metrics = train_cv_models(df, 'alpha', config)
    
    # 保存α模型 / Save α models
    for i, model in enumerate(alpha_models):
        model_file = config.MODEL_OUTPUT_DIR / f'alpha_model_fold{i+1}.pkl'
        joblib.dump(model, model_file)
    
    print(f"  保存了 {len(alpha_models)} 个α模型")
    print(f"  Saved {len(alpha_models)} α models")
    
    # 步骤3: 训练Qb,p模型 / Step 3: Train Qb,p models
    print("\n[步骤3/4] 训练Qb,p参数的BRT模型...")
    print("[Step 3/4] Training BRT models for Qb,p parameter...")
    
    qbp_models, qbp_metrics = train_cv_models(df, 'Qbp_mm_yr', config)
    
    # 保存Qb,p模型 / Save Qb,p models
    for i, model in enumerate(qbp_models):
        model_file = config.MODEL_OUTPUT_DIR / f'qbp_model_fold{i+1}.pkl'
        joblib.dump(model, model_file)
    
    print(f"  保存了 {len(qbp_models)} 个Qb,p模型")
    print(f"  Saved {len(qbp_models)} Qb,p models")
    
    # 步骤4: 保存性能指标 / Step 4: Save performance metrics
    print("\n[步骤4/4] 保存性能指标...")
    print("[Step 4/4] Saving performance metrics...")
    
    # 合并性能指标 / Combine metrics
    all_metrics = pd.concat([alpha_metrics, qbp_metrics])
    metrics_file = config.MODEL_OUTPUT_DIR / 'model_performance_metrics.csv'
    all_metrics.to_csv(metrics_file, index=False)
    
    print(f"  性能指标已保存到: {metrics_file}")
    print(f"  Performance metrics saved to: {metrics_file}")
    
    print("\n" + "="*80)
    print("模型训练完成! / Model training completed!")
    print("="*80)
    
    # 打印最终摘要 / Print final summary
    print("\n最终性能摘要 / Final Performance Summary:")
    print("\nα参数模型 / α Parameter Models:")
    print(f"  平均测试 R²: {alpha_metrics['test_r2'].mean():.4f}")
    print(f"  平均测试 RMSE: {alpha_metrics['test_rmse'].mean():.4f}")
    
    print("\nQb,p参数模型 / Qb,p Parameter Models:")
    print(f"  平均测试 R²: {qbp_metrics['test_r2'].mean():.4f}")
    print(f"  平均测试 RMSE: {qbp_metrics['test_rmse'].mean():.4f} mm/yr")


if __name__ == "__main__":
    main()
