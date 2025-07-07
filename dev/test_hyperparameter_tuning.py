# test_hyperparameter_tuning.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from hyperparameter_tuning import HyperparameterTuner
from env import StockTradingEnv
from rl_agent import PPOAgent
from evaluation import calculate_metrics, calculate_benchmark_returns

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 设置路径
DATA_DIR = "data"
RESULTS_DIR = "results/hyperparameter_tuning"
MODEL_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """加载并准备数据"""
    # 读取带有特征的股票数据
    data_file = os.path.join(DATA_DIR, "stock_data_with_features.csv")
    print(f"读取特征数据: {data_file}")
    data = pd.read_csv(data_file)
    
    # 转换日期列为日期时间类型并排序
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['date', 'tic'])
    
    # 获取交易股票列表
    ticker_list = data['tic'].unique().tolist()
    
    # 划分训练集和测试集 (时间顺序，前70%训练，后30%测试)
    unique_dates = data['date'].unique()
    train_split_index = int(len(unique_dates) * 0.7)
    train_dates = unique_dates[:train_split_index]
    test_dates = unique_dates[train_split_index:]
    
    train_data = data[data['date'].isin(train_dates)]
    test_data = data[data['date'].isin(test_dates)]
    
    return train_data, test_data, ticker_list

def create_environments(train_data, test_data, ticker_list, config):
    """创建训练和测试环境"""
    # 从配置中获取环境参数
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    
    # 创建训练环境
    train_env = StockTradingEnv(
        df=train_data,
        stock_dim=len(ticker_list),
        hmax=10,
        initial_amount=config.get('initial_amount', 1000000),
        transaction_cost_pct=config.get('transaction_cost_pct', 0.001),
        reward_scaling=config.get('reward_scaling', 1e-4),
        tech_indicator_list=tech_indicators
    )
    
    # 创建测试环境
    test_env = StockTradingEnv(
        df=test_data,
        stock_dim=len(ticker_list),
        hmax=10,
        initial_amount=config.get('initial_amount', 1000000),
        transaction_cost_pct=config.get('transaction_cost_pct', 0.001),
        reward_scaling=config.get('reward_scaling', 1e-4),
        tech_indicator_list=tech_indicators
    )
    
    return train_env, test_env

def train_function(config):
    """训练函数，用于超参数调优"""
    print("开始训练，配置:", config)
    
    # 加载数据
    train_data, _, ticker_list = load_data()
    
    # 技术指标列表
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    
    # 创建训练环境
    train_env = StockTradingEnv(
        df=train_data,
        stock_dim=len(ticker_list),
        hmax=10,
        initial_amount=config.get('initial_amount', 1000000),
        transaction_cost_pct=config.get('transaction_cost_pct', 0.001),
        reward_scaling=config.get('reward_scaling', 1e-4),
        tech_indicator_list=tech_indicators
    )
    
    # 创建代理
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config.get('learning_rate', 0.0003),
        gamma=config.get('gamma', 0.99),
        clip_range=config.get('clip_range', 0.2),
        batch_size=config.get('batch_size', 64),
        n_epochs=config.get('n_epochs', 10),
        gae_lambda=config.get('gae_lambda', 0.95),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        ent_coef=config.get('ent_coef', 0.01),
        vf_coef=config.get('vf_coef', 0.5),
        net_arch=config.get('network_arch', [64, 64]),
        activation_fn=config.get('activation_fn', 'tanh')
    )
    
    # 训练代理
    n_episodes = config.get('n_episodes', 20)  # 为测试目的减少轮次
    max_steps = config.get('max_steps', 1000)
    
    # 训练历史记录
    episode_rewards = []
    episode_returns = []
    
    # 训练循环
    for episode in range(1, n_episodes + 1):
        state = train_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, _ = train_env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            if agent.memory.size >= agent.batch_size:
                agent.update()
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # 记录这一轮的奖励
        episode_rewards.append(episode_reward)
        portfolio_value = train_env.portfolio_value
        initial_value = train_env.initial_amount
        episode_return = (portfolio_value - initial_value) / initial_value
        episode_returns.append(episode_return)
        
        if episode % 5 == 0:
            print(f"Episode {episode}/{n_episodes}, Reward: {episode_reward:.4f}, Return: {episode_return:.4f}")
    
    # 计算训练指标
    train_metrics = {
        "total_reward": sum(episode_rewards),
        "mean_reward": np.mean(episode_rewards),
        "final_return": episode_returns[-1],
        "max_return": max(episode_returns),
        "sharpe_ratio": np.mean(episode_returns) / (np.std(episode_returns) + 1e-8),
        "max_drawdown": calculate_max_drawdown(episode_returns)
    }
    
    return train_metrics, episode_rewards, agent

def evaluate_function(agent, config):
    """评估函数，用于超参数调优"""
    print("开始评估...")
    
    # 加载数据
    _, test_data, ticker_list = load_data()
    
    # 技术指标列表
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    
    # 创建测试环境
    test_env = StockTradingEnv(
        df=test_data,
        stock_dim=len(ticker_list),
        hmax=10,
        initial_amount=config.get('initial_amount', 1000000),
        transaction_cost_pct=config.get('transaction_cost_pct', 0.001),
        reward_scaling=config.get('reward_scaling', 1e-4),
        tech_indicator_list=tech_indicators
    )
    
    # 设置评估模式
    agent.training = False
    
    # 执行测试
    state = test_env.reset()
    done = False
    portfolio_values = [test_env.portfolio_value]
    weights_history = []
    total_reward = 0
    
    while not done:
        # 记录当前权重
        weights_history.append(test_env.weights.copy())
        
        # 选择动作
        action = agent.select_action(state)
        next_state, reward, done, _ = test_env.step(action)
        
        # 更新状态和奖励
        state = next_state
        total_reward += reward
        
        # 记录投资组合价值
        portfolio_values.append(test_env.portfolio_value)
    
    # 计算评估指标
    initial_value = test_env.initial_amount
    final_value = portfolio_values[-1]
    cumulative_return = (final_value - initial_value) / initial_value
    
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)  # 年化夏普比率
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    test_metrics = {
        "total_reward": total_reward,
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_portfolio_value": final_value
    }
    
    return test_metrics, portfolio_values, weights_history

def compare_function(agent, config):
    """比较函数，与基准策略对比"""
    print("开始与基准策略比较...")
    
    # 加载数据
    _, test_data, ticker_list = load_data()
    
    # 计算基准策略收益
    # 这里使用简单的等权重策略作为基准
    benchmark_returns = calculate_benchmark_returns(test_data, ticker_list)
    
    # 执行RL策略测试
    test_metrics, portfolio_values, _ = evaluate_function(agent, config)
    rl_return = test_metrics["cumulative_return"]
    
    # 比较指标
    comparison_metrics = {
        "vs_equal_weight": rl_return - benchmark_returns.get("equal_weight", 0),
        "vs_price_inverse": rl_return - benchmark_returns.get("price_inverse", 0),
        "vs_momentum": rl_return - benchmark_returns.get("momentum", 0)
    }
    
    return comparison_metrics

def calculate_max_drawdown(values):
    """计算最大回撤"""
    values = np.array(values)
    max_so_far = values[0]
    max_drawdown = 0
    
    for value in values:
        if value > max_so_far:
            max_so_far = value
        drawdown = (max_so_far - value) / max_so_far
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

def run_test():
    """运行超参数调优测试"""
    print("开始超参数调优测试...")
    
    # 基础配置
    base_config = {
        # 环境参数
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        
        # 训练参数
        "n_episodes": 10,  # 为测试减少轮次
        "max_steps": 500,
        
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
    
    # 创建调优器
    tuner = HyperparameterTuner(base_config, "test_run")
    
    # 定义参数网格 (简单测试，只使用少量值)
    param_grid = {
        "learning_rate": [0.0001, 0.0003],
        "gamma": [0.95, 0.99],
        "batch_size": [32, 64]
    }
    
    # 运行网格搜索
    print("开始网格搜索...")
    best_exp_id = tuner.run_grid_search(
        param_grid=param_grid,
        train_func=train_function,
        evaluate_func=evaluate_function,
        compare_func=compare_function
    )
    
    print(f"调优完成! 最佳实验ID: {best_exp_id}")
    
    # 获取最佳配置
    best_config = tuner.tracker.get_best_experiment()["config"]
    print("最佳配置:")
    for key, value in best_config.items():
        if key in param_grid:
            print(f"  {key}: {value}")
    
    return best_exp_id, tuner

if __name__ == "__main__":
    run_test()