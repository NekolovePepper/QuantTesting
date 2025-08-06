# Quant Testing: Portfolio-selection

This project aims to develop an adaptive financial trading strategy based on a Mixture-of-Experts (MoE) reinforcement learning architecture with help of vibe coding, supporting multiple baseline strategies and deep RL methods. It represents a self‑guided learning exploration by a beginner in constructing a reinforcement learning framework tailored for quant trading.

## Directory Structure

- `data/`： raw and feature‑engineered stock data
- `dev/`：development and testing scripts
- `docs/`：project documentation and progress notes
- `logs/`：training and evaluation logs
- `models/`：saved model weights
- `results/`：backtest outcomes and evaluation reports
- `src/`：core source code, including submodules for baseline strategies, RL agents, plotting, core logic, evaluation, hyperparameters, utilities, etc.
- 
## Start

1. Install dependencies：
   ```bash
   conda env create -f environment.yml
   conda activate QuantTesting
   ```
2. Prepare data and run feature engineering
   ```bash
   python src/download_data.py
   python src/feature_engineering.py
   ```
3. Evaluate baseline strategies：
   ```bash
   python src/main.py strategy
   ```
4. Train and evaluate RL agent：
   ```bash
   python src/main.py train
   python src/main.py evaluate
   ```


