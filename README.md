🌀 Budyko-Analysis | Hydrological water-energy balance for 6000+ Chinese catchments  
🌀 Budyko-Analysis | 覆盖 6000+ 中国流域的水文能量平衡分析框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 One-liner | 项目一句话
Analyze catchment water-energy balance with Budyko theory + LAI/CO2-aware PET + deviation/trajectory analytics, built for large-scale China hydrology.  
通过 Budyko 理论、支持 LAI/CO2 的 PET 估算，以及偏差/轨迹分析，实现中国大尺度流域水能平衡研究。

## 🧰 Tech Stack | 技术栈
- Python 3.8+, NumPy/Pandas/SciPy, xarray, scikit-learn, matplotlib  
- Python 3.8+，NumPy/Pandas/SciPy，xarray，scikit-learn，matplotlib
- Parallelism via `multiprocessing` + tqdm; notebooks and scripts for workflows  
- 通过 `multiprocessing`+tqdm 并行；提供 notebook 与脚本工作流

## 🗂️ File Structure | 目录结构
- `main_budyko_workflow.py`: end-to-end demo of PET comparison, Budyko indices, deviation/trajectory, attribution.  
  `main_budyko_workflow.py`: 端到端演示，含 PET 对比、Budyko 指数、偏差/轨迹与归因。
- `src/` (SSOT core)
  - `budyko/`: theory ops — `curves.py` (Fu/Tixeront), `water_balance.py` (EA= P-Q, IA/IE), `deviation.py`, `trajectory_jaramillo.py`.  
    `budyko/`: 理论核心——曲线、水平衡、偏差与轨迹。
  - `models/`: PET engines — `pet_models.py` (classical), `pet_lai_co2.py` (LAI+CO2 innovation).  
    `models/`: PET 计算——传统方法与 LAI+CO2 创新。
  - `data_processing/`: ingestion/QC — `basin_processor.py` (runoff & met extraction), `cmip6_processor.py`, `grace_lai_processor.py`.  
    `data_processing/`: 数据加载与质控——径流/气象、CMIP6、GRACE/LAI。
  - `analysis/`: higher-level analytics — `budyko_ml_workflow.py`, `deviation_attribution.py`, `snow_analyzer.py`.  
    `analysis/`: 高阶分析——Budyko 约束机器学习、偏差归因、积雪分析。
  - `utils/parallel_processing.py`: `ParallelBudykoAnalyzer` for thousands of catchments.  
    `utils/parallel_processing.py`: 并行处理万级流域。
  - `visualization/`: `budyko_plots.py`, `direction_rose.py`.  
    `visualization/`: Budyko 空间与方向玫瑰可视化。
- `examples/`: runnable guides (`01_real_data_workflow.py`, `complete_workflow_example.py`, notebooks).  
  `examples/`: 可运行示例（真实数据工作流、完整工作流、notebook）。
- `tests/`: unit/integration tests for PET, water balance, full workflow.  
  `tests/`: PET、水量平衡、全流程的单元与集成测试。
- `docs/`, `notebooks/`, `Scripts/`, `results/`, `outputs/`: supporting docs, tutorials, batch scripts, sample outputs.  
  `docs/`、`notebooks/`、`Scripts/`、`results/`、`outputs/`: 文档、教程、批处理脚本与示例结果。

## 🔑 Key Source Code | 核心代码导航
- Entry workflow: `main_budyko_workflow.py` — orchestrates PET calc → Budyko indices → deviation/trajectory → attribution/plots.  
  入口工作流：`main_budyko_workflow.py` — 串联 PET、Budyko 指标、偏差/轨迹与归因/可视化。
- Budyko theory: `src/budyko/curves.py` (Fu/Tixeront curve, ω fitting); `water_balance.py` (EA=P-Q, IA/IE, QC).  
  Budyko 理论：`src/budyko/curves.py`（曲线与 ω 拟合）；`water_balance.py`（EA=P-Q，IA/IE，质控）。
- PET core: `src/models/pet_lai_co2.py` (LAI+CO2 Penman-Monteith variant); `pet_models.py` (classics).  
  PET 核心：`src/models/pet_lai_co2.py`（LAI+CO2 版 PM）；`pet_models.py`（传统集合）。
- Data pipeline: `src/data_processing/basin_processor.py` (runoff loading, gridded extraction, aggregation/QC).  
  数据管线：`src/data_processing/basin_processor.py`（径流加载、格点提取、聚合/质控）。
- Advanced analysis: `src/analysis/deviation_attribution.py`, `trajectory_jaramillo.py`, `analysis/budyko_ml_workflow.py`.  
  高阶分析：`src/analysis/deviation_attribution.py`、`trajectory_jaramillo.py`、`analysis/budyko_ml_workflow.py`。
- Scaling: `src/utils/parallel_processing.py` — safe parallel executor with error capture.  
  扩展：`src/utils/parallel_processing.py` — 带错误收集的并行执行。

## 🧭 Code Walkthrough Path | 源码阅读路径
1) Start with `examples/01_real_data_workflow.py` — see end-to-end usage & inputs/outputs.  
   从 `examples/01_real_data_workflow.py` 入手，整体感受输入输出。
2) Open `src/data_processing/basin_processor.py` — how runoff (Q) & met data are loaded, QC’d, aggregated.  
   阅读 `basin_processor.py`，理解径流/气象加载、质控与聚合。
3) Read `src/models/pet_lai_co2.py` & `pet_models.py` — PET calculation pathways (innovation vs baseline).  
   查看 `pet_lai_co2.py` 与 `pet_models.py`，区分创新与基线 PET。
4) Read `src/budyko/water_balance.py` → `src/budyko/curves.py` — compute IA/IE from P,Q,PET then fit ω.  
   阅读 `water_balance.py`→`curves.py`，理解 IA/IE 计算与 ω 拟合。
5) Explore `src/budyko/deviation.py` & `trajectory_jaramillo.py` — deviation stats & movement vectors.  
   探索 `deviation.py` 与 `trajectory_jaramillo.py`，掌握偏差统计与轨迹向量。
6) Inspect `src/analysis/deviation_attribution.py` & `analysis/budyko_ml_workflow.py` — attribution & Budyko-constrained ML.  
   查看 `deviation_attribution.py` 与 `analysis/budyko_ml_workflow.py`，了解归因与约束式 ML。
7) For scale-out, read `src/utils/parallel_processing.py` — how tasks are chunked and validated.  
   需要扩展时，阅读 `parallel_processing.py`，掌握任务切分与结果校验。
8) Finally, check `main_budyko_workflow.py` orchestration and `visualization/` for plotting.  
   最后回到 `main_budyko_workflow.py` 与 `visualization/`，理解调度与可视化。

## 🔄 Data Flow | 数据流转
Runoff/forcing ingestion (`basin_processor`) → PET calc (`pet_lai_co2` / `pet_models`) → Water balance IA/IE (`water_balance`) → Curve fitting & deviation/trajectory (`curves`, `deviation`, `trajectory_jaramillo`) → Attribution/ML (`analysis/*`) → Parallel scaling (`utils/parallel_processing`) → Plots (`visualization/*`) → Outputs `outputs/`, `results/`.  
数据链路：径流/气象加载 → PET 计算 → IA/IE 水量平衡 → 曲线拟合与偏差/轨迹 → 归因/ML → 并行扩展 → 可视化输出。

## 🧭 Real-world Mapping | 业务场景映射
- `BasinDataProcessor`: ingest/QC runoff & meteorology → “数据基座/观测锚点”。  
  `BasinDataProcessor`：加载并质控径流和气象，现实中的观测基础。
- `WaterBalanceCalculator`: computes EA=P-Q and indices → “水量收支核算”。  
  `WaterBalanceCalculator`：计算实际蒸发与指标，相当于收支表。
- `BudykoCurves`: ω fitting & theoretical IE → “理论基准线/健康曲线”。  
  `BudykoCurves`：拟合流域参数，形成理论参照。
- `PETWithLAICO2`: PET with vegetation & CO2 response → “植被-大气耦合蒸发需求”。  
  `PETWithLAICO2`：考虑植被与CO2响应的蒸散需求。
- `DeviationAnalysis` / `DeviationAttribution`: quantify & explain departures → “异常诊断与病因分析”。  
  `DeviationAnalysis` / `DeviationAttribution`：偏差诊断与驱动归因。
- `TrajectoryAnalyzer`: movement in Budyko space → “演化轨迹/方向玫瑰”。  
  `TrajectoryAnalyzer`：捕捉流域响应方向与强度。
- `ParallelBudykoAnalyzer`: batch 1000s catchments → “大规模批处理引擎”。  
  `ParallelBudykoAnalyzer`：面向大批量流域的并行执行。

## 🚀 Quickstart | 快速开始
```bash
git clone https://github.com/yourusername/Budyko-Analysis.git
cd Budyko-Analysis
python -m venv .venv && .\.venv\Scripts\activate  # Windows 示例
pip install -r requirements.txt
```
Run demo 示例:
```bash
python examples/complete_workflow_example.py
```
Outputs go to `outputs/complete_workflow/`.  
输出保存在 `outputs/complete_workflow/`。

## 🧪 Tests | 测试
```bash
pytest tests/ -v
```
Or target units/integration separately.  
可分别运行单测或集成测试。

## 📚 Reading Order for New Devs | 新同事阅读顺序
1) `README` (本文件) + `QUICKSTART.md` → high-level intent.  
   先读 `README` 与 `QUICKSTART.md`，把握全局。
2) `examples/01_real_data_workflow.py` → concrete usage.  
   看真实工作流脚本，理解输入输出格式。
3) `src/data_processing/basin_processor.py` → data/QC contracts.  
   深入数据契约和质控。
4) `src/models/pet_lai_co2.py` & `src/budyko/water_balance.py` → PET & IA/IE core.  
   理解 PET 计算与 IA/IE 生成。
5) `src/budyko/curves.py`, `src/budyko/deviation.py`, `trajectory_jaramillo.py` → theory/diagnostics.  
   研读曲线、偏差、轨迹。
6) `src/analysis/deviation_attribution.py`, `analysis/budyko_ml_workflow.py` → attribution/ML.  
   了解归因与 ML 扩展。
7) `src/utils/parallel_processing.py` → scaling patterns.  
   熟悉并行模式与错误处理。

## 🛠️ Common Entry Points | 常用入口
- Minimal PET+WB:
  ```python
  from src.models.pet_lai_co2 import PETWithLAICO2
  from src.budyko.water_balance import WaterBalanceCalculator
  pet = PETWithLAICO2().calculate(temperature=T, humidity=RH, wind_speed=U2, radiation=Rn, lai=LAI, co2=CO2)  # mm/day
  wb = WaterBalanceCalculator().calculate_budyko_indices(P=P, Q=Q, PET=pet * 365)
  ```
  最小示例：计算 PET，再用径流 Q 得到 IA/IE。
- Parallel batch:
  ```python
  from src.utils.parallel_processing import ParallelBudykoAnalyzer
  analyzer = ParallelBudykoAnalyzer(n_processes=8)
  df = analyzer.process_catchments(catchment_ids, analysis_function=my_fn, data_loader=my_loader)
  ```
  并行处理多流域，捕获失败详情。

## 📈 Performance Tips | 性能提示
- Use array/vectorized PET (`pet_lai_co2.py`) and avoid per-year Python loops.  
  使用向量化 PET，避免逐年循环。
- Set `n_processes` wisely (CPU-1) and moderate `chunk_size`.  
  `n_processes` 设为 CPU-1，`chunk_size` 适度。
- Warm-start ω with climate-based guess (`fit_omega` smart_guess).  
  利用 `fit_omega` 的智能初值减少迭代。

## 📦 Data Requirements | 数据要求
- Mandatory: runoff Q (mm/day or mm/month), precipitation P, meteorological drivers for PET.  
  必需：径流 Q、降水 P、气象驱动（温湿风辐射）用于 PET。
- Optional: LAI (MODIS), CO2, GRACE TWS, land-use/irrigation/reservoir drivers for attribution.  
  可选：LAI、CO2、GRACE TWS 及土地利用/灌溉/水库等归因因子。
- Default folders: place processed inputs under `data/processed/`, outputs under `results/` or `outputs/`.  
  默认目录：输入放 `data/processed/`，输出在 `results/` 或 `outputs/`。

## 🤝 Contribution | 贡献
- Fork → feature branch → tests/docs → PR.  
  Fork → 新分支 → 补充测试/文档 → 提 PR。
- Style: PEP8, docstrings, add/extend tests under `tests/`.  
  规范：PEP8、完善注释，补充 `tests/`。

## 📜 License | 许可证
MIT License, see `LICENSE`.  
MIT 许可证，详见 `LICENSE`。

## 📬 Contact | 联系方式
- Issues on GitHub; email placeholder `your.email@example.com`.  
- GitHub Issues；邮件 `your.email@example.com`。

## ✅ Newcomer Checklist | 新人自查
- [ ] Can load runoff (Q) + PET to compute IA/IE via `water_balance`.  
  [ ] 能用径流与 PET 计算 IA/IE。
- [ ] Can fit ω and quantify deviation/trajectory for two periods.  
  [ ] 会拟合 ω 并计算时段偏差/轨迹。
- [ ] Can swap PET methods (baseline vs LAI+CO2) and compare IE/ε.  
  [ ] 会切换 PET 方法并比较 IE/偏差。
- [ ] Can run `examples/01_real_data_workflow.py` and read outputs in `outputs/`.  
  [ ] 能运行示例并查看输出。
- [ ] Know how to batch with `ParallelBudykoAnalyzer` for many catchments.  
  [ ] 掌握并行批处理用法。

---

**祝研究顺利！如有问题，请随时提出Issue。**
