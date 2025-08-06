# Quant Testing: Portfolio-selection

本项目目标实现基于混合专家（MoE）强化学习的自适应金融交易策略，支持多种基线策略与深度强化学习方法。
是初学者纯自学背景下的一次探索。

## 目录结构

- `data/`：原始及特征工程后的股票数据
- `dev/`：开发与测试脚本
- `docs/`：项目文档与进展
- `logs/`：训练与评估日志
- `models/`：保存的模型权重
- `results/`：回测与评估结果
- `src/`：核心源代码（含 baseline、rl、plot、core、evaluation、hyperparameter、utils 等子模块）
## 快速开始

1. 安装依赖：
   ```bash
   conda env create -f environment.yml
   conda activate QuantTesting
   ```
2. 数据准备与特征工程：
   ```bash
   python src/download_data.py
   python src/feature_engineering.py
   ```
3. 运行基线策略评估：
   ```bash
   python src/main.py strategy
   ```
4. 训练与评估 RL 代理：
   ```bash
   python src/main.py train
   python src/main.py evaluate
   ```


