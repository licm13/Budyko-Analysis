# 快速入门指南 (Quick Start Guide)

## 项目概述 (Project Overview)

本项目复现了论文 "Global Runoff Partitioning Based on Budyko-Constrained Machine Learning" (Cheng et al., 2025, Water Resources Research) 的核心方法和结果。

This project replicates the core methods and results from the paper "Global Runoff Partitioning Based on Budyko-Constrained Machine Learning" (Cheng et al., 2025, Water Resources Research).

---

## 核心创新 (Core Innovation)

该研究的创新点在于:  
The innovation of this study is:

1. **物理约束的机器学习**: 将Budyko物理框架与BRT机器学习结合
2. **全球径流分割**: 首次提供全球尺度的基流和快流分离数据集
3. **驱动因素识别**: 使用SHAP方法识别控制径流分割的主要因子

1. **Physically-constrained ML**: Combining Budyko physical framework with BRT machine learning
2. **Global runoff partitioning**: First global-scale baseflow and quickflow separation dataset
3. **Driver identification**: Using SHAP to identify main factors controlling runoff partitioning

---

## 5分钟快速上手 (5-Minute Quick Start)

### 1. 克隆并设置环境 (Clone and Setup Environment)

```bash
# 克隆仓库 / Clone repository
git clone <repository_url>
cd paper_replication_budyko_ml_runoff

# 创建虚拟环境 / Create virtual environment
conda create -n budyko_ml python=3.9
conda activate budyko_ml

# 安装依赖 / Install dependencies
pip install -r requirements.txt
```

### 2. 准备数据 (Prepare Data)

由于数据文件较大,请按照以下步骤下载:  
Due to large file sizes, please download data following these steps:

```bash
# 创建数据目录 / Create data directories
mkdir -p data/raw/discharge/GRDC
mkdir -p data/raw/discharge/CAMELS
mkdir -p data/raw/meteorology
mkdir -p data/raw/catchment_properties
```

**下载清单** (Download Checklist):
- [ ] GRDC流量数据 / GRDC discharge data → `data/raw/discharge/GRDC/`
- [ ] CAMELS流量数据 / CAMELS discharge data → `data/raw/discharge/CAMELS/`
- [ ] MSWEP降水数据 / MSWEP precipitation → `data/raw/meteorology/`
- [ ] TerraClimate潜在蒸散发 / TerraClimate PET → `data/raw/meteorology/`
- [ ] GLEAM实际蒸散发 / GLEAM AET → `data/raw/meteorology/`
- [ ] 13个流域属性数据 / 13 catchment properties → `data/raw/catchment_properties/`

详细数据源链接见主README.md / See main README.md for detailed data source links

### 3. 运行工作流 (Run Workflow)

```bash
# 方式1: 运行完整工作流 / Option 1: Run full workflow
bash scripts/run_full_workflow.sh

# 方式2: 逐步运行 / Option 2: Run step by step
cd src
python 01_data_preprocessing.py      # 数据预处理 / Data preprocessing
python 02_parameter_estimation.py    # 参数估计 / Parameter estimation  
python 03_model_training.py          # 模型训练 / Model training
python 04_global_prediction.py       # 全球预测 / Global prediction
python 05_visualization.py           # 可视化 / Visualization
python 06_driver_analysis.py         # 驱动因素分析 / Driver analysis
```

### 4. 查看结果 (View Results)

```bash
# 启动Jupyter / Start Jupyter
jupyter notebook

# 打开以下notebook / Open these notebooks:
# - notebooks/exploratory_analysis.ipynb
# - notebooks/results_verification.ipynb
```

---

## 项目结构说明 (Project Structure Explanation)

```
paper_replication_budyko_ml_runoff/
│
├── README.md                    # 主文档 / Main documentation
├── QUICK_START.md               # 本文件 / This file
├── requirements.txt             # Python依赖 / Python dependencies
├── .gitignore                   # Git忽略文件 / Git ignore
│
├── data/                        # 数据目录 / Data directory
│   ├── raw/                     # 原始数据 / Raw data (需下载)
│   ├── processed/               # 处理后数据 / Processed data
│   └── external/                # 外部辅助数据 / External data
│
├── src/                         # 源代码 / Source code
│   ├── 01_data_preprocessing.py      # 步骤1 / Step 1
│   ├── 02_parameter_estimation.py    # 步骤2 / Step 2
│   ├── 03_model_training.py          # 步骤3 / Step 3
│   ├── 04_global_prediction.py       # 步骤4 / Step 4
│   ├── 05_visualization.py           # 步骤5 / Step 5
│   ├── 06_driver_analysis.py         # 步骤6 / Step 6
│   └── utils.py                      # 工具函数 / Utilities
│
├── notebooks/                   # Jupyter笔记本 / Jupyter notebooks
├── results/                     # 结果输出 / Results output
│   ├── figures/                 # 图表 / Figures
│   ├── tables/                  # 表格 / Tables
│   └── models/                  # 模型 / Models
│
├── docs/                        # 文档 / Documentation
│   └── methodology_details.md   # 方法详解 / Detailed methods
│
└── scripts/                     # 脚本 / Scripts
    └── run_full_workflow.sh     # 完整工作流 / Full workflow
```

---

## 关键文件说明 (Key Files Explanation)

### 核心Python模块 (Core Python Modules)

| 文件 | 功能 | 输入 | 输出 |
|-----|------|------|------|
| `utils.py` | 基础函数库 | - | Budyko/BFC方程,评估指标 |
| `01_data_preprocessing.py` | 数据预处理 | 原始流量/气象数据 | 长期平均值,基流分离 |
| `02_parameter_estimation.py` | 参数估计 | 处理后数据 | α和Qb,p参数 |
| `03_model_training.py` | 模型训练 | 参数+流域属性 | 训练的BRT模型 |
| `04_global_prediction.py` | 全球预测 | 训练模型+全球数据 | 全球径流分割 |
| `05_visualization.py` | 可视化 | 预测结果 | 图表 |
| `06_driver_analysis.py` | 驱动分析 | BRT模型 | SHAP重要性 |

### 关键方程 (Key Equations)

**Budyko曲线 (径流):**
```
Q/P = -Ep/P + [1 + (Ep/P)^α]^(1/α)
```

**BFC曲线 (基流):**
```
Qb/P = Qbp/P + [1+(Ep/P)^α]^(1/α) - [1+(Ep/P+Qbp/P)^α]^(1/α)
```

**快流 (快流):**
```
Qq = Q - Qb
```

---

## 预期运行时间 (Expected Runtime)

在标准工作站上 (8核CPU, 32GB RAM):  
On a standard workstation (8-core CPU, 32GB RAM):

| 步骤 | 预期时间 | 备注 |
|-----|---------|------|
| 数据预处理 | 1-2小时 | 取决于流域数量 |
| 参数估计 | 10-20分钟 | 迭代求解 |
| 模型训练 | 30-60分钟 | 10折交叉验证 |
| 全球预测 | 2-3小时 | 全球网格计算 |
| 可视化 | 10-20分钟 | 生成图表 |
| 驱动分析 | 30-60分钟 | SHAP计算 |
| **总计** | **约5-8小时** | 首次运行 |

---

## 故障排除 (Troubleshooting)

### 常见问题 (Common Issues)

**1. 内存不足 (Out of Memory)**
```bash
# 解决方案: 减少批处理大小或使用Dask
# Solution: Reduce batch size or use Dask
```

**2. 缺少数据文件 (Missing Data Files)**
```bash
# 检查数据目录 / Check data directory
ls -la data/raw/
# 确保已下载所有必需数据 / Ensure all required data is downloaded
```

**3. 包导入错误 (Package Import Errors)**
```bash
# 重新安装依赖 / Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**4. CUDA错误 (XGBoost with GPU)**
```bash
# 如果不使用GPU,修改代码中的 tree_method
# If not using GPU, modify tree_method in code
# 将 'gpu_hist' 改为 'hist'
```

---

## 下一步 (Next Steps)

完成快速入门后,您可以:  
After completing the quick start, you can:

1. **探索结果** / Explore results:
   - 查看Jupyter notebooks中的交互式分析
   - View interactive analysis in Jupyter notebooks

2. **修改参数** / Modify parameters:
   - 调整BRT超参数提高性能
   - Adjust BRT hyperparameters to improve performance

3. **扩展方法** / Extend methods:
   - 添加新的流域属性
   - Add new catchment properties
   - 尝试其他ML算法
   - Try other ML algorithms

4. **应用到新区域** / Apply to new regions:
   - 使用您自己的流域数据
   - Use your own catchment data

---

## 获取帮助 (Getting Help)

- **Issues**: https://github.com/yourusername/repo/issues
- **Email**: your.email@example.com
- **原始论文** / Original paper: Cheng et al. (2025), DOI: 10.1029/2025WR039863

---

## 引用 (Citation)

如果使用本代码,请引用:  
If you use this code, please cite:

```bibtex
@article{cheng2025global,
  title={Global Runoff Partitioning Based on Budyko-Constrained Machine Learning},
  author={Cheng, Shujie and Hulsman, Petra and Koppa, Akash and Beck, Hylke E. and Xia, Jun and Xu, Jijun and Cheng, Lei and Miralles, Diego G.},
  journal={Water Resources Research},
  volume={61},
  pages={e2025WR039863},
  year={2025},
  doi={10.1029/2025WR039863}
}
```

---

**祝您复现顺利! / Good luck with your replication!** 🚀
