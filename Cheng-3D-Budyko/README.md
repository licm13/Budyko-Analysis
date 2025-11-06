# Global Runoff Partitioning Based on Budyko-Constrained Machine Learning
# 基于Budyko约束机器学习的全球径流分割

## 论文信息 (Paper Information)
* **期刊 (Journal):** Water Resources Research, 2025
* **DOI:** 10.1029/2025WR039863
* **一句话总结 (Summary):** 开发了一种结合Budyko框架和机器学习(BRT)的混合方法,用于全球尺度径流分割(基流和快流)估算
* **关键词 (Keywords):** Budyko framework, Machine Learning, Baseflow, Runoff Partitioning, Global Hydrology

---

## 复现说明 (Replication Note)
本代码库旨在最大程度地复现上述论文的分析方法和核心结果。  
This repository aims to (to the best extent possible) replicate the analysis methodology and core findings of the aforementioned paper.

* **复现状态 (Replication Status):** 已完成核心方法实现 / Core methodology implemented
* **主要差异 (Known Deviations):** 
  - 使用公开的流域数据(可能与原文数据集略有不同)
  - 部分辅助数据已更新到最新版本
  - Uses publicly available catchment data (may differ slightly from original dataset)
  - Some auxiliary data updated to latest versions

---

## 核心方法与数学描述 (Core Methodology & Mathematical Description)

本文使用 Budyko 框架结合机器学习进行径流归因分析。  
This paper uses the Budyko framework combined with machine learning for runoff attribution analysis.

### 关键模型 (Key Models)

**1. Budyko曲线 (Budyko Curve) - 径流估算**

$$\frac{Q}{P} = 1 - \frac{E_a}{P} = -\frac{E_p}{P} + \left[1 + \left(\frac{E_p}{P}\right)^\alpha\right]^{1/\alpha}$$

其中 / Where:
- Q: 径流 / Runoff
- P: 降水 / Precipitation
- Ea: 实际蒸散发 / Actual Evaporation
- Ep: 潜在蒸散发 / Potential Evaporation
- α: Budyko参数,反映流域特性 / Budyko parameter reflecting catchment properties

**2. BFC曲线 (Baseflow Coefficient Curve) - 基流估算**

$$\frac{Q_b}{P} = \frac{Q_{b,p}}{P} + \left[1 + \left(\frac{E_p}{P}\right)^\alpha\right]^{1/\alpha} - \left[1 + \left(\frac{E_p}{P} + \frac{Q_{b,p}}{P}\right)^\alpha\right]^{1/\alpha}$$

其中 / Where:
- Qb: 基流 / Baseflow
- Qb,p: 潜在基流 / Potential Baseflow
- α: 与Budyko曲线共享的参数 / Shared parameter with Budyko curve

**3. 快流 (Quickflow)**

$$Q_q = Q - Q_b$$

详细的数学推导和公式说明请查看 `docs/methodology_details.md`。  
(See `docs/methodology_details.md` for detailed mathematical formulas and derivations.)

---

## 分析工作流 (Analysis Workflow)

### 1. 数据预处理 (Data Preprocessing)
* **输入 (Input)**: `data/raw/` - 原始流量、气象、流域属性数据
* **脚本 (Script)**: `src/01_data_preprocessing.py`
* **输出 (Output)**: `data/processed/` - 处理后的数据,包括:
  - 长期平均P, Ep, Ea, Q
  - 基流分离后的 Qb, Qq
  - 流域属性数据集

### 2. 参数估计 (Parameter Estimation)
* **输入 (Input)**: `data/processed/` 的处理数据
* **脚本 (Script)**: `src/02_parameter_estimation.py`
* **输出 (Output)**: 
  - 流域尺度的 α 参数 (从Budyko曲线)
  - 流域尺度的 Qb,p 参数 (从BFC曲线)

### 3. 机器学习模型训练 (Model Training)
* **输入 (Input)**: 参数(α, Qb,p) + 13个流域属性
* **脚本 (Script)**: `src/03_model_training.py`
* **输出 (Output)**: 
  - 训练好的BRT模型 (10折交叉验证)
  - 模型性能指标 (R², RMSE)

### 4. 全球预测 (Global Prediction)
* **输入 (Input)**: 全球网格的P, Ep和流域属性
* **脚本 (Script)**: `src/04_global_prediction.py`
* **输出 (Output)**: 
  - 全球0.25°网格的 α, Qb,p
  - 全球径流 Q, 基流 Qb, 快流 Qq
  - 相应的系数 (RFC, BFC, QFC, BFI, QFI)

### 5. 结果可视化 (Visualization)
* **输入 (Input)**: 模型输出和全球预测结果
* **脚本 (Script)**: `src/05_visualization.py`
* **输出 (Output)**: `results/figures/` - 复现论文中的关键图表

### 6. 驱动因素分析 (Driver Analysis)
* **输入 (Input)**: 训练的BRT模型
* **脚本 (Script)**: `src/06_driver_analysis.py`
* **方法 (Method)**: SHAP (Shapley Additive Explanations)
* **输出 (Output)**: 各流域的主导驱动因素空间分布

---

## 数据 (Data)

### 观测数据 (Observational Data)
| 数据名称 | 来源 | 分辨率 | 时段 | 访问 |
|---------|------|--------|------|------|
| 流量数据 | GRDC | 流域尺度 | 1980-2020 | https://www.bafg.de/GRDC/ |
| 流量数据 | CAMELS (US, Chile, Brazil, UK, Australia) | 流域尺度 | 各国不同 | 见各CAMELS数据集链接 |

### 气象强迫数据 (Meteorological Forcing)
| 数据名称 | 来源 | 分辨率 | 时段 | 访问 |
|---------|------|--------|------|------|
| 降水 (P) | MSWEP v1.1 | 0.25° | 1980-2020 | Beck et al. 2017 |
| 潜在蒸散发 (Ep) | TerraClimate | 1/24° → 0.25° | 1980-2020 | https://www.climatologylab.org/terraclimate.html |
| 实际蒸散发 (Ea) | GLEAM v3.6 | 0.25° | 1980-2020 | https://www.gleam.eu/ |

### 流域属性数据 (Catchment Properties)

#### 气候属性 (Climate)
| 变量 | 来源 | 分辨率 | 访问 |
|-----|------|--------|------|
| 气温 (TC) | ERA5 | 0.25° | Copernicus Climate Data Store |
| 季节性指数 (SAI) | 从P和Ep计算 | 0.25° | Liu et al. 2018 |
| 雪水当量 (SWE) | GLOBSNOW + NSIDC | 0.25° | NSIDC |

#### 植被属性 (Vegetation)
| 变量 | 来源 | 分辨率 | 访问 |
|-----|------|--------|------|
| NDVI | MODIS | 0.05° → 0.25° | https://modis.gsfc.nasa.gov/ |
| WUE | MODIS | 0.05° → 0.25° | https://modis.gsfc.nasa.gov/ |
| LAI | MODIS | 0.05° → 0.25° | https://modis.gsfc.nasa.gov/ |
| 根深 (RD) | Fan et al. 2017 | ~1 km → 0.25° | PNAS 2017 |

#### 地形属性 (Topography)
| 变量 | 来源 | 分辨率 | 访问 |
|-----|------|--------|------|
| 地形指数 (CTI) | Marthews et al. 2015 | 500 m → 0.25° | HESS 2015 |
| 海拔 (ELEV) | MERIT DEM | 90 m → 0.25° | Yamazaki et al. 2019 |
| 坡度 (SLO) | Amatulli et al. 2018 | 1 km → 0.25° | Scientific Data 2018 |

#### 土壤属性 (Soil)
| 变量 | 来源 | 分辨率 | 访问 |
|-----|------|--------|------|
| 土壤厚度 (STHI) | Pelletier et al. 2016 | 1 km → 0.25° | Science 2016 |
| 孔隙度 (SPO) | SoilGrids 2.0 | 250 m → 0.25° | https://soilgrids.org/ |

#### 人类活动 (Human Activities)
| 变量 | 来源 | 分辨率 | 访问 |
|-----|------|--------|------|
| 人类影响指数 (HFP) | Sanderson et al. 2002 | 1 km → 0.25° | BioScience 2002 |

**注意 (Note)**: 
- 由于原始数据的分辨率不同,所有数据均使用双线性插值重采样到0.25°
- All data resampled to 0.25° using bilinear interpolation due to varying original resolutions
- 原始数据需用户自行下载并放置到 `data/raw/` 目录
- Users need to download raw data and place in `data/raw/` directory

---

## 如何运行 (How to Run)

### 1. 克隆仓库 (Clone Repository)
```bash
git clone https://github.com/yourusername/paper_replication_budyko_ml_runoff.git
cd paper_replication_budyko_ml_runoff
```

### 2. 创建环境并安装依赖 (Create Environment & Install Dependencies)
```bash
# 使用 conda 创建环境
# Create environment using conda
conda create -n budyko_ml python=3.9
conda activate budyko_ml

# 安装依赖
# Install dependencies
pip install -r requirements.txt
```

### 3. 下载数据 (Download Data)
**重要 (Important)**: 由于数据量巨大,本仓库不包含原始数据。请按照以下步骤获取数据:

1. **流量数据 (Discharge Data)**:
   - GRDC: 注册后从 https://www.bafg.de/GRDC/ 下载
   - CAMELS: 从各国CAMELS项目网站下载

2. **气象数据 (Meteorological Data)**:
   - MSWEP: 从 http://www.gloh2o.org/mswep/ 下载
   - TerraClimate: 从 https://www.climatologylab.org/terraclimate.html 下载
   - GLEAM: 从 https://www.gleam.eu/ 下载
   - ERA5: 从 Copernicus Climate Data Store 下载

3. **流域属性 (Catchment Properties)**:
   - 详见上方数据表中的链接
   - See links in data table above

将下载的数据放置在相应的 `data/raw/` 子目录中。  
Place downloaded data in corresponding `data/raw/` subdirectories.

### 4. 执行工作流 (Run Workflow)
```bash
# 步骤 1: 预处理数据
# Step 1: Preprocess data
python src/01_data_preprocessing.py

# 步骤 2: 估计流域参数
# Step 2: Estimate catchment parameters
python src/02_parameter_estimation.py

# 步骤 3: 训练机器学习模型
# Step 3: Train machine learning models
python src/03_model_training.py

# 步骤 4: 全球预测
# Step 4: Global prediction
python src/04_global_prediction.py

# 步骤 5: 生成可视化结果
# Step 5: Generate visualizations
python src/05_visualization.py

# 步骤 6: 驱动因素分析
# Step 6: Driver analysis
python src/06_driver_analysis.py
```

或者运行完整流程:  
Or run the complete workflow:
```bash
bash scripts/run_full_workflow.sh
```

### 5. 探索性分析 (Exploratory Analysis)
使用Jupyter Notebooks进行交互式探索:  
Use Jupyter Notebooks for interactive exploration:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 项目结构 (Project Structure)
```
paper_replication_budyko_ml_runoff/
│
├── README.md                          # 本文件 / This file
├── requirements.txt                   # Python依赖 / Python dependencies
├── .gitignore                        # Git忽略文件 / Git ignore file
│
├── data/
│   ├── raw/                          # 原始数据 (用户需下载) / Raw data (user download)
│   │   ├── discharge/                # 流量数据 / Discharge data
│   │   ├── meteorology/              # 气象数据 / Meteorological data
│   │   ├── catchment_properties/     # 流域属性 / Catchment properties
│   │   └── .gitkeep
│   ├── processed/                    # 处理后的数据 / Processed data
│   │   └── .gitkeep
│   └── external/                     # 外部辅助数据 / External auxiliary data
│       └── .gitkeep
│
├── src/                              # 源代码 / Source code
│   ├── __init__.py
│   ├── 01_data_preprocessing.py      # 数据预处理 / Data preprocessing
│   ├── 02_parameter_estimation.py    # 参数估计 / Parameter estimation
│   ├── 03_model_training.py          # 模型训练 / Model training
│   ├── 04_global_prediction.py       # 全球预测 / Global prediction
│   ├── 05_visualization.py           # 可视化 / Visualization
│   ├── 06_driver_analysis.py         # 驱动因素分析 / Driver analysis
│   └── utils.py                      # 辅助函数 / Utility functions
│
├── notebooks/                        # Jupyter Notebooks
│   ├── exploratory_analysis.ipynb    # 探索性分析 / Exploratory analysis
│   └── results_verification.ipynb    # 结果验证 / Results verification
│
├── results/
│   ├── figures/                      # 生成的图表 / Generated figures
│   │   └── .gitkeep
│   ├── tables/                       # 生成的表格 / Generated tables
│   │   └── .gitkeep
│   └── models/                       # 保存的模型 / Saved models
│       └── .gitkeep
│
├── docs/
│   └── methodology_details.md        # 方法论详细说明 / Detailed methodology
│
└── scripts/
    └── run_full_workflow.sh          # 完整工作流脚本 / Full workflow script
```

---

## 主要结果 (Key Results)

基于论文的主要发现:  
Based on key findings from the paper:

1. **全球径流分割 (Global Runoff Partitioning)**:
   - 全球平均径流: 292 mm/yr (占降水的35.3%)
   - 基流: 162 mm/yr (占降水的19.6%)
   - 快流: 130 mm/yr (占降水的15.7%)
   - Global average runoff: 292 mm/yr (35.3% of precipitation)
   - Baseflow: 162 mm/yr (19.6% of precipitation)
   - Quickflow: 130 mm/yr (15.7% of precipitation)

2. **模型性能 (Model Performance)**:
   - 径流估算: R² = 0.96, RMSE = 51 mm/yr (测试集)
   - 基流估算: R² = 0.91, RMSE = 49 mm/yr (测试集)
   - Runoff estimation: R² = 0.96, RMSE = 51 mm/yr (test set)
   - Baseflow estimation: R² = 0.91, RMSE = 49 mm/yr (test set)

3. **主要驱动因素 (Primary Drivers)**:
   - 参数 α: 主要受植被属性控制(42.4%流域),其中海拔(ELEV)最重要(31.4%)
   - 参数 Qb,p: 主要受土壤孔隙度(SPO)控制(52.2%流域)
   - Parameter α: Primarily controlled by vegetation properties (42.4% of catchments), with elevation (ELEV) being most important (31.4%)
   - Parameter Qb,p: Predominantly governed by soil porosity (SPO) (52.2% of catchments)

---

## 引用 (Citation)

如果您使用了本代码库,请引用原始论文:  
If you use this code, please cite the original paper:

```bibtex
@article{cheng2025global,
  title={Global Runoff Partitioning Based on Budyko-Constrained Machine Learning},
  author={Cheng, Shujie and Hulsman, Petra and Koppa, Akash and Beck, Hylke E. and Xia, Jun and Xu, Jijun and Cheng, Lei and Miralles, Diego G.},
  journal={Water Resources Research},
  volume={61},
  pages={e2025WR039863},
  year={2025},
  publisher={Wiley Online Library},
  doi={10.1029/2025WR039863}
}
```

---

## 许可证 (License)
本项目遵循 MIT 许可证 - 详见 LICENSE 文件  
This project is licensed under the MIT License - see the LICENSE file for details

---

## 联系方式 (Contact)
如有问题或建议,请联系:  
For questions or suggestions, please contact:

- **Repository**: https://github.com/yourusername/paper_replication_budyko_ml_runoff
- **Issues**: https://github.com/yourusername/paper_replication_budyko_ml_runoff/issues

---

## 致谢 (Acknowledgments)
本复现工作基于 Cheng et al. (2025) 的原始研究。感谢原作者提供的方法论和数据描述。  
This replication work is based on the original research by Cheng et al. (2025). We thank the authors for their methodology and data descriptions.

数据来源致谢见上方数据表。  
Data source acknowledgments are listed in the data table above.
