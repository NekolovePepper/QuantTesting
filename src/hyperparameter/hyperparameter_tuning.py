import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
import copy
import csv

# 设置结果保存目录
TUNING_RESULTS_DIR = "results/hyperparameter_tuning"
os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)

class ExperimentTracker:
    """
    实验跟踪器类，用于管理强化学习调参实验
    记录参数设置、训练过程和评估结果
    """
    def __init__(self, experiment_name: str = None):
        """
        初始化实验跟踪器
        
        参数:
            experiment_name: 实验名称，如果为None则使用时间戳
        """
        if experiment_name is None:
            self.experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.experiment_name = experiment_name
            
        # 创建实验目录
        self.experiment_dir = os.path.join(TUNING_RESULTS_DIR, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 初始化实验记录
        self.experiments = []
        self.best_experiment = None
        self.metrics_log_path = os.path.join(self.experiment_dir, "experiments_log.csv")
        
        # 创建CSV日志文件头
        self._create_log_file()
    
    def _create_log_file(self):
        """创建实验日志文件"""
        with open(self.metrics_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                "实验ID", "时间戳", 
                # 环境参数
                "初始资金", "交易成本", "奖励缩放", 
                # RL参数
                "算法", "学习率", "批次大小", "轮数", "GAE_LAMBDA",
                "GAMMA", "CLIP_RANGE", "值函数系数", "熵系数",
                "网络结构", "激活函数",
                # 训练指标
                "训练集累计奖励", "训练集平均奖励", "训练集夏普比率", "训练集最大回撤",
                # 测试指标
                "测试集累计奖励", "测试集累计回报率", "测试集夏普比率", "测试集最大回撤", 
                # 比较指标
                "相对等权重策略收益", "相对价格反向策略收益", "相对动量策略收益", 
                # 额外信息
                "备注"
            ]
            writer.writerow(header)
    
    def log_experiment(self, 
                      config: Dict[str, Any], 
                      train_metrics: Dict[str, float],
                      test_metrics: Dict[str, float],
                      comparison_metrics: Dict[str, float] = None,
                      episode_rewards: List[float] = None,
                      portfolio_values: List[float] = None,
                      weights_history: List[np.ndarray] = None,
                      notes: str = None) -> str:
        """
        记录一次实验结果
        
        参数:
            config: 参数配置字典
            train_metrics: 训练集评估指标
            test_metrics: 测试集评估指标
            comparison_metrics: 与基线策略的比较指标
            episode_rewards: 每个训练轮次的奖励列表
            portfolio_values: 投资组合价值历史
            weights_history: 投资组合权重历史
            notes: 实验备注
            
        返回:
            experiment_id: 实验ID
        """
        # 生成实验ID
        experiment_id = f"exp_{len(self.experiments) + 1}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建实验记录
        experiment = {
            "id": experiment_id,
            "timestamp": timestamp,
            "config": copy.deepcopy(config),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "comparison_metrics": comparison_metrics or {},
            "notes": notes or ""
        }
        
        # 添加到实验列表
        self.experiments.append(experiment)
        
        # 检查是否是最佳实验（根据测试集夏普比率）
        if self.best_experiment is None or \
           test_metrics.get("sharpe_ratio", 0) > self.best_experiment["test_metrics"].get("sharpe_ratio", 0):
            self.best_experiment = experiment
            
        # 将实验结果写入CSV日志
        with open(self.metrics_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                experiment_id, timestamp,
                # 环境参数
                config.get("initial_amount", ""),
                config.get("transaction_cost_pct", ""),
                config.get("reward_scaling", ""),
                # RL参数
                config.get("algorithm", ""),
                config.get("learning_rate", ""),
                config.get("batch_size", ""),
                config.get("n_epochs", ""),
                config.get("gae_lambda", ""),
                config.get("gamma", ""),
                config.get("clip_range", ""),
                config.get("vf_coef", ""),
                config.get("ent_coef", ""),
                config.get("network_arch", ""),
                config.get("activation_fn", ""),
                # 训练指标
                train_metrics.get("total_reward", ""),
                train_metrics.get("mean_reward", ""),
                train_metrics.get("sharpe_ratio", ""),
                train_metrics.get("max_drawdown", ""),
                # 测试指标
                test_metrics.get("total_reward", ""),
                test_metrics.get("cumulative_return", ""),
                test_metrics.get("sharpe_ratio", ""),
                test_metrics.get("max_drawdown", ""),
                # 比较指标
                comparison_metrics.get("vs_equal_weight", ""),
                comparison_metrics.get("vs_price_inverse", ""),
                comparison_metrics.get("vs_momentum", ""),
                # 备注
                notes or ""
            ]
            writer.writerow(row)
            
        # 保存详细信息
        self._save_experiment_details(experiment_id, experiment, episode_rewards, portfolio_values, weights_history)
        
        # 更新可视化
        self.visualize_experiments()
        
        return experiment_id
    
    def _save_experiment_details(self, 
                               experiment_id: str, 
                               experiment: Dict[str, Any],
                               episode_rewards: List[float] = None,
                               portfolio_values: List[float] = None,
                               weights_history: List[np.ndarray] = None):
        """保存实验的详细信息和数据"""
        # 创建实验详细信息目录
        experiment_detail_dir = os.path.join(self.experiment_dir, experiment_id)
        os.makedirs(experiment_detail_dir, exist_ok=True)
        
        # 保存配置和指标
        with open(os.path.join(experiment_detail_dir, 'config.json'), 'w') as f:
            json.dump({k: str(v) for k, v in experiment["config"].items()}, f, indent=4)
            
        with open(os.path.join(experiment_detail_dir, 'metrics.json'), 'w') as f:
            metrics = {
                "train_metrics": experiment["train_metrics"],
                "test_metrics": experiment["test_metrics"],
                "comparison_metrics": experiment["comparison_metrics"]
            }
            json.dump({k: {k2: str(v2) for k2, v2 in v.items()} for k, v in metrics.items()}, f, indent=4)
        
        # 保存训练奖励历史
        if episode_rewards is not None:
            rewards_df = pd.DataFrame({
                "episode": list(range(1, len(episode_rewards) + 1)),
                "reward": episode_rewards
            })
            rewards_df.to_csv(os.path.join(experiment_detail_dir, 'episode_rewards.csv'), index=False)
            
            # 绘制奖励曲线
            plt.figure(figsize=(10, 6))
            plt.plot(rewards_df["episode"], rewards_df["reward"])
            plt.title(f"training reward curve - {experiment_id}")
            plt.xlabel("epochs")
            plt.ylabel("reward")
            plt.grid(True)
            plt.savefig(os.path.join(experiment_detail_dir, 'reward_curve.png'))
            plt.close()
        
        # 保存投资组合价值历史
        if portfolio_values is not None:
            portfolio_df = pd.DataFrame({
                "day": list(range(1, len(portfolio_values) + 1)),
                "value": portfolio_values
            })
            portfolio_df.to_csv(os.path.join(experiment_detail_dir, 'portfolio_values.csv'), index=False)
            
            # 绘制投资组合价值曲线
            plt.figure(figsize=(10, 6))
            plt.plot(portfolio_df["day"], portfolio_df["value"])
            plt.title(f"Portfolio value curve - {experiment_id}")
            plt.xlabel("Trading Day")
            plt.ylabel("Portfolio value")
            plt.grid(True)
            plt.savefig(os.path.join(experiment_detail_dir, 'portfolio_value_curve.png'))
            plt.close()
            
            # 计算并绘制收益率曲线
            portfolio_df["return"] = portfolio_df["value"] / portfolio_df["value"].iloc[0] - 1
            plt.figure(figsize=(10, 6))
            plt.plot(portfolio_df["day"], portfolio_df["return"])
            plt.title(f"accumulated return rate curve- {experiment_id}")
            plt.xlabel("Trading Day")
            plt.ylabel("accumulated return rate")
            plt.grid(True)
            plt.savefig(os.path.join(experiment_detail_dir, 'return_curve.png'))
            plt.close()
        
        # 保存权重历史
        if weights_history is not None:
            # 转换为DataFrame
            weights_data = np.array(weights_history)
            n_assets = weights_data.shape[1]
            weights_df = pd.DataFrame(
                weights_data, 
                columns=[f"asset_{i+1}" for i in range(n_assets)]
            )
            weights_df["day"] = list(range(1, len(weights_df) + 1))
            weights_df.to_csv(os.path.join(experiment_detail_dir, 'weights_history.csv'), index=False)
            
            # 绘制权重随时间变化图
            plt.figure(figsize=(12, 8))
            for i in range(n_assets):
                plt.plot(weights_df["day"], weights_df[f"asset_{i+1}"], label=f"资产 {i+1}")
            # 投资组合权重随时间变化
            plt.title(f"Portfolio weights over time - {experiment_id}") 
            plt.xlabel("Trading Day")
            plt.ylabel("Weight")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(experiment_detail_dir, 'weights_over_time.png'))
            plt.close()
            
            # 绘制平均权重饼图
            avg_weights = weights_df.iloc[:, :-1].mean().values
            plt.figure(figsize=(10, 10))
            plt.pie(
                avg_weights, 
                labels=[f"资产 {i+1}" for i in range(n_assets)],
                autopct='%1.1f%%'
            )
            plt.title(f"平均资产权重分布 - {experiment_id}")
            plt.savefig(os.path.join(experiment_detail_dir, 'avg_weights_pie.png'))
            plt.close()
    
    def visualize_experiments(self):
        """生成所有实验的比较可视化"""
        if len(self.experiments) < 1:
            return
            
        # 创建比较数据框
        comparison_data = []
        for exp in self.experiments:
            row = {
                "实验ID": exp["id"],
                "时间戳": exp["timestamp"]
            }
            
            # 添加关键参数
            for param_key in ["learning_rate", "batch_size", "n_epochs", "gamma", "clip_range"]:
                if param_key in exp["config"]:
                    row[param_key] = exp["config"][param_key]
            
            # 添加关键指标
            row["训练奖励"] = exp["train_metrics"].get("total_reward", 0)
            row["测试奖励"] = exp["test_metrics"].get("total_reward", 0)
            row["测试回报率"] = exp["test_metrics"].get("cumulative_return", 0)
            row["测试夏普比率"] = exp["test_metrics"].get("sharpe_ratio", 0)
            row["测试最大回撤"] = exp["test_metrics"].get("max_drawdown", 0)
            
            comparison_data.append(row)
        
        # 转换为DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较表格
        comparison_df.to_csv(os.path.join(self.experiment_dir, 'experiments_comparison.csv'), index=False)
        
        # 绘制关键指标比较图
        for metric in ["测试奖励", "测试回报率", "测试夏普比率", "测试最大回撤"]:
            if metric in comparison_df.columns:
                plt.figure(figsize=(12, 6))
                ax = sns.barplot(x="实验ID", y=metric, data=comparison_df)
                plt.title(f"实验比较 - {metric}")
                plt.xticks(rotation=45)
                
                # 添加数值标签
                for i, p in enumerate(ax.patches):
                    ax.annotate(f"{p.get_height():.4f}", 
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha = 'center', va = 'bottom',
                                rotation=45)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.experiment_dir, f'comparison_{metric.replace(" ", "_")}.png'))
                plt.close()
        
        # 绘制参数与性能关系散点图
        for param in ["learning_rate", "batch_size", "n_epochs", "gamma"]:
            if param in comparison_df.columns:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=param, y="测试夏普比率", data=comparison_df, s=100)
                
                # 添加标签
                for i, row in comparison_df.iterrows():
                    if param in row and "测试夏普比率" in row:
                        plt.text(row[param], row["测试夏普比率"], row["实验ID"], 
                                fontsize=9)
                
                plt.title(f"{param} 与夏普比率关系")
                plt.grid(True)
                plt.savefig(os.path.join(self.experiment_dir, f'param_impact_{param}.png'))
                plt.close()

    def get_best_experiment(self) -> Dict[str, Any]:
        """获取最佳实验配置和结果"""
        return self.best_experiment
    
    def load_from_csv(self, csv_path: str):
        """从CSV文件加载实验记录"""
        if not os.path.exists(csv_path):
            print(f"CSV文件不存在: {csv_path}")
            return
            
        experiments_df = pd.read_csv(csv_path)
        for _, row in experiments_df.iterrows():
            experiment_id = row["实验ID"]
            experiment_dir = os.path.join(self.experiment_dir, experiment_id)
            
            if os.path.exists(experiment_dir):
                # 读取配置
                with open(os.path.join(experiment_dir, 'config.json'), 'r') as f:
                    config = json.load(f)
                
                # 读取指标
                with open(os.path.join(experiment_dir, 'metrics.json'), 'r') as f:
                    metrics = json.load(f)
                
                # 创建实验记录
                experiment = {
                    "id": experiment_id,
                    "timestamp": row["时间戳"],
                    "config": config,
                    "train_metrics": metrics["train_metrics"],
                    "test_metrics": metrics["test_metrics"],
                    "comparison_metrics": metrics.get("comparison_metrics", {}),
                    "notes": row.get("备注", "")
                }
                
                # 添加到实验列表
                self.experiments.append(experiment)
                
                # 更新最佳实验
                if self.best_experiment is None or \
                   metrics["test_metrics"].get("sharpe_ratio", 0) > \
                   self.best_experiment["test_metrics"].get("sharpe_ratio", 0):
                    self.best_experiment = experiment

class HyperparameterTuner:
    """
    超参数调优类，用于管理不同的参数配置并执行实验
    """
    def __init__(self, base_config: Dict[str, Any], experiment_name: str = None):
        """
        初始化超参数调优器
        
        参数:
            base_config: 基础参数配置
            experiment_name: 实验名称
        """
        self.base_config = base_config
        self.tracker = ExperimentTracker(experiment_name)
        
    def generate_configs(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        根据参数网格生成配置列表
        
        参数:
            param_grid: 参数网格，每个参数对应一个取值列表
            
        返回:
            配置列表
        """
        import itertools
        
        # 提取参数名和取值列表
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # 生成参数组合
        configs = []
        for values in itertools.product(*param_values):
            # 基于基础配置创建新配置
            config = copy.deepcopy(self.base_config)
            
            # 更新参数
            for name, value in zip(param_names, values):
                # 处理嵌套参数 (例如: "policy.learning_rate")
                if "." in name:
                    parts = name.split(".")
                    target = config
                    for part in parts[:-1]:
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    target[parts[-1]] = value
                else:
                    config[name] = value
            
            configs.append(config)
        
        return configs
    
    def run_experiment(self, 
                      config: Dict[str, Any], 
                      train_func, 
                      evaluate_func,
                      compare_func = None,
                      notes: str = None) -> str:
        """
        运行一次实验
        
        参数:
            config: 参数配置
            train_func: 训练函数，接收配置返回(train_metrics, episode_rewards, agent)
            evaluate_func: 评估函数，接收agent和配置返回(test_metrics, portfolio_values, weights_history)
            compare_func: 比较函数，接收agent和配置返回comparison_metrics
            notes: 实验备注
            
        返回:
            experiment_id: 实验ID
        """
        # 训练代理
        train_metrics, episode_rewards, agent = train_func(config)
        
        # 评估代理
        test_metrics, portfolio_values, weights_history = evaluate_func(agent, config)
        
        # 与基线策略比较
        comparison_metrics = {}
        if compare_func is not None:
            comparison_metrics = compare_func(agent, config)
        
        # 记录实验
        experiment_id = self.tracker.log_experiment(
            config=config,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            comparison_metrics=comparison_metrics,
            episode_rewards=episode_rewards,
            portfolio_values=portfolio_values,
            weights_history=weights_history,
            notes=notes
        )
        
        return experiment_id
    
    def run_grid_search(self, 
                       param_grid: Dict[str, List[Any]], 
                       train_func, 
                       evaluate_func,
                       compare_func = None) -> str:
        """
        执行网格搜索
        
        参数:
            param_grid: 参数网格
            train_func: 训练函数
            evaluate_func: 评估函数
            compare_func: 比较函数
            
        返回:
            best_experiment_id: 最佳实验ID
        """
        # 生成配置列表
        configs = self.generate_configs(param_grid)
        
        print(f"将执行 {len(configs)} 次实验进行超参数搜索")
        
        # 执行实验
        for i, config in enumerate(configs):
            print(f"执行实验 {i+1}/{len(configs)}")
            
            # 生成实验备注，显示当前参数组合
            param_desc = ", ".join([f"{k}={v}" for k, v in config.items() 
                                   if k in param_grid or any(k.startswith(p+".") for p in param_grid)])
            notes = f"网格搜索实验 #{i+1}: {param_desc}"
            
            # 运行实验
            experiment_id = self.run_experiment(
                config=config,
                train_func=train_func,
                evaluate_func=evaluate_func,
                compare_func=compare_func,
                notes=notes
            )
            
            print(f"实验 {experiment_id} 完成")
        
        # 获取最佳实验
        best_experiment = self.tracker.get_best_experiment()
        if best_experiment:
            best_config = best_experiment["config"]
            best_metrics = best_experiment["test_metrics"]
            
            print("\n===== 网格搜索结果 =====")
            print(f"最佳实验ID: {best_experiment['id']}")
            print(f"最佳参数组合: {json.dumps({k: v for k, v in best_config.items() if k in param_grid or any(k.startswith(p+'.') for p in param_grid)}, indent=2)}")
            print(f"测试集表现: 夏普比率={best_metrics.get('sharpe_ratio', 0):.4f}, 累计回报率={best_metrics.get('cumulative_return', 0):.4f}")
            
            return best_experiment['id']
        
        return None

# 使用示例
if __name__ == "__main__":
    print("超参数调优框架已创建")
    print(f"调参结果将保存在: {TUNING_RESULTS_DIR}")
    
    # 示例：设置基础配置
    base_config = {
        # 环境参数
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        
        # RL参数
        "algorithm": "PPO",
        "learning_rate": 0.0003,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "network_arch": [64, 64],
        "activation_fn": "tanh"
    }
    
    print("\n你可以这样使用超参数调优框架:")
    print("1. 创建调优器: tuner = HyperparameterTuner(base_config, '实验名称')")
    print("2. 定义参数网格: param_grid = {'learning_rate': [0.0001, 0.0003], 'gamma': [0.95, 0.99]}")
    print("3. 运行网格搜索: tuner.run_grid_search(param_grid, train_func, evaluate_func, compare_func)")
    print("4. 查看最佳参数: best_config = tuner.tracker.get_best_experiment()['config']")