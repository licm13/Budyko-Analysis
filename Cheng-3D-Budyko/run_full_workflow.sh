#!/bin/bash

# 完整工作流脚本 (Full Workflow Script)
# 运行Budyko-ML径流分割项目的所有步骤
# Run all steps of the Budyko-ML runoff partitioning project

echo "================================================================================"
echo "Budyko-ML 径流分割完整工作流"
echo "Budyko-ML Runoff Partitioning Full Workflow"
echo "================================================================================"
echo ""

# 设置错误时退出 / Exit on error
set -e

# 切换到src目录 / Change to src directory
cd "$(dirname "$0")/../src"

echo "[步骤 1/6] 数据预处理 / Data Preprocessing"
echo "----------------------------------------------------------------------"
python 01_data_preprocessing.py
if [ $? -ne 0 ]; then
    echo "错误: 数据预处理失败 / Error: Data preprocessing failed"
    exit 1
fi
echo ""

echo "[步骤 2/6] 参数估计 / Parameter Estimation"
echo "----------------------------------------------------------------------"
python 02_parameter_estimation.py
if [ $? -ne 0 ]; then
    echo "错误: 参数估计失败 / Error: Parameter estimation failed"
    exit 1
fi
echo ""

echo "[步骤 3/6] 模型训练 / Model Training"
echo "----------------------------------------------------------------------"
python 03_model_training.py
if [ $? -ne 0 ]; then
    echo "错误: 模型训练失败 / Error: Model training failed"
    exit 1
fi
echo ""

echo "[步骤 4/6] 全球预测 / Global Prediction"
echo "----------------------------------------------------------------------"
python 04_global_prediction.py
if [ $? -ne 0 ]; then
    echo "错误: 全球预测失败 / Error: Global prediction failed"
    exit 1
fi
echo ""

echo "[步骤 5/6] 结果可视化 / Visualization"
echo "----------------------------------------------------------------------"
python 05_visualization.py
if [ $? -ne 0 ]; then
    echo "错误: 可视化失败 / Error: Visualization failed"
    exit 1
fi
echo ""

echo "[步骤 6/6] 驱动因素分析 / Driver Analysis"
echo "----------------------------------------------------------------------"
python 06_driver_analysis.py
if [ $? -ne 0 ]; then
    echo "错误: 驱动因素分析失败 / Error: Driver analysis failed"
    exit 1
fi
echo ""

echo "================================================================================"
echo "工作流完成! / Workflow completed!"
echo "================================================================================"
echo ""
echo "输出位置 / Output locations:"
echo "  - 处理后的数据 / Processed data: ../data/processed/"
echo "  - 训练的模型 / Trained models: ../results/models/"
echo "  - 生成的图表 / Generated figures: ../results/figures/"
echo "  - 生成的表格 / Generated tables: ../results/tables/"
echo ""
echo "下一步 / Next steps:"
echo "  1. 查看结果 / Review results: jupyter notebook ../notebooks/results_verification.ipynb"
echo "  2. 生成报告 / Generate report: (可添加自定义报告生成脚本)"
echo ""
