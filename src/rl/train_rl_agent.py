import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from src.core.env import StockTradingEnv
from src.rl.PPO import PPOAgent
from src.rl.A2C import A2CAgent


from src.evaluation.evaluation import (
    calculate_metrics,
    calculate_benchmark_returns,
    plot_portfolio_values,
    plot_returns_comparison,
    plot_benchmark_comparison,
    create_metrics_table
)
# 导入增强型奖励环境
from src.core.enhanced_env import EnhancedRewardStockTradingEnv

# 设置路径
DATA_DIR = "data"
RESULTS_DIR = "results/rl_agent"
MODEL_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 设置随机种子以提高可重现性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


def train_agent(agent, env, n_episodes=100, max_steps=1000, save_freq=10):
    """
    训练RL代理
    
    参数:
        agent: RL代理
        env: 训练环境
        n_episodes (int): 训练轮数
        max_steps (int): 每轮最大步数
        save_freq (int): 保存频率 (每save_freq个轮次保存一次模型)
    """
    # 训练历史记录
    training_history = {
        'episode_rewards': [],
        'episode_returns': [],
        'episode_steps': []
    }
    
    # 设置训练模式
    agent.training = True
    
    print("开始训练RL代理...")
    for episode in range(1, n_episodes + 1):
        # 重置环境
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        portfolio_values = [env.portfolio_value]
        actions_history = []
        # 执行一个episode
        for _ in range(max_steps):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储转移数据
            agent.store_transition(state, action, reward, next_state, done)
            
            # 累计奖励
            episode_reward += reward
            portfolio_values.append(env.portfolio_value)

            # 更新状态
            state = next_state
            step_count += 1
            actions_history.append(action)
            if done:
                break
        
        # 训练模型
        agent.train()
        print(portfolio_values[-1])
        # 计算回报率
        
        episode_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # 记录训练数据
        training_history['episode_rewards'].append(episode_reward)
        training_history['episode_returns'].append(episode_return)
        training_history['episode_steps'].append(step_count)
        training_history['actions_history'] = actions_history
        # 打印训练进度
        print(f"Episode {episode}/{n_episodes} - 回报: {episode_reward:.4f}, "
              f"回报率: {episode_return:.4f}, 步数: {step_count}")
        
        # 定期保存模型
        if episode % save_freq == 0:
            agent.save(MODEL_DIR, f"ppo_agent_episode_{episode}")
    
    # 绘制训练曲线
    plot_training_curves(training_history)
    
    # 保存最终模型
    agent.save(MODEL_DIR, "ppo_agent_final")
    
    print("RL代理训练完成!")
    return training_history

def evaluate_agent(agent, env, ticker_list, test_data):
    """
    评估RL代理性能
    
    参数:
        agent: RL代理
        env: 测试环境
        ticker_list: 股票代码列表
        test_data: 测试数据
    
    返回:
        dict: 包含评估结果的字典
    """
    # 设置评估模式
    agent.training = False
    
    # 重置环境
    state = env.reset()
    done = False
    episode_reward = 0
    portfolio_values = [env.portfolio_value]
    actions_history = []
    
    print("开始评估RL代理...")
    
    # 执行一个episode
    while not done:
        # 选择动作
        action = agent.select_action(state)
        
        # 记录动作
        actions_history.append(action)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 累计奖励
        episode_reward += reward
        portfolio_values.append(env.portfolio_value)
        
        # 更新状态
        state = next_state
    
    # 计算性能指标
    metrics = calculate_metrics(portfolio_values)
    
    # 打印评估结果
    print("\nRL代理评估结果:")
    print(f"累计回报率: {metrics['cumulative_return']:.4f}")
    print(f"年化收益率: {metrics['annual_return']:.4f}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"索提诺比率: {metrics['sortino_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f}")
    print(f"波动率: {metrics['volatility']:.4f}")
    print(f"胜率: {metrics['win_rate']:.4f}")
    
    # 计算基准收益
    benchmark_returns = calculate_benchmark_returns(test_data, ticker_list)
    
    # 分析投资组合权重变化
    analyze_portfolio_weights(actions_history, ticker_list)
    
    # 返回评估结果
    return {
        "name": "PPO代理",
        "metrics": metrics,
        "portfolio_values": portfolio_values,
        "total_reward": episode_reward,
        "actions_history": actions_history
    }

def plot_training_curves(training_history):
    """
    绘制训练曲线
    
    参数:
        training_history (dict): 训练历史记录
    """
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制累积奖励曲线
    episodes = range(1, len(training_history['episode_rewards']) + 1)
    ax1.plot(episodes, training_history['episode_rewards'], 'b-')
    ax1.set_title('累积奖励 vs. 训练轮次')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('累积奖励')
    ax1.grid(True)
    
    # 绘制回报率曲线
    ax2.plot(episodes, training_history['episode_returns'], 'r-')
    ax2.set_title('回报率 vs. 训练轮次')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('回报率')
    ax2.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'))
    plt.close()

def analyze_portfolio_weights(actions_history, ticker_list):
    """
    分析投资组合权重变化
    
    参数:
        actions_history (list): 动作历史
        ticker_list (list): 股票代码列表
    """
    actions_array = np.array(actions_history)
    
    # 计算平均权重
    avg_weights = np.mean(actions_array, axis=0)
    
    # 计算权重标准差 (衡量权重波动性)
    std_weights = np.std(actions_array, axis=0)
    
    # 创建权重分析表格
    weight_data = {
        'Stock': ticker_list,
        'Avg Weight': avg_weights,
        'Std Dev': std_weights
    }
    weight_df = pd.DataFrame(weight_data)
    
    # 打印权重分析
    print("\n投资组合权重分析:")
    print(weight_df)
    
    # 保存权重分析
    weight_df.to_csv(os.path.join(RESULTS_DIR, 'portfolio_weights.csv'), index=False)
    
    # 绘制权重变化图
    plt.figure(figsize=(12, 6))
    for i in range(len(ticker_list)):
        plt.plot(actions_array[:, i], label=ticker_list[i])
    
    plt.title('投资组合权重随时间变化')
    plt.xlabel('交易日')
    plt.ylabel('权重')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'weights_over_time.png'))
    plt.close()
    
    # 绘制平均权重饼图
    plt.figure(figsize=(10, 10))
    plt.pie(avg_weights, labels=ticker_list, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # 确保饼图是圆的
    plt.title('平均投资组合权重分布')
    plt.savefig(os.path.join(RESULTS_DIR, 'avg_weights_pie.png'))
    plt.close()

def compare_with_baselines(rl_results, test_env, ticker_list, tech_indicators, test_data):
    """
    与基线策略比较
    
    参数:
        rl_results (dict): RL代理评估结果
        test_env: 测试环境
        ticker_list: 股票代码列表
        tech_indicators: 技术指标列表
        test_data: 测试数据
    """
    # 导入基线策略
    from src.core.strategies import (
        equal_weight_strategy,
        price_inverse_strategy,
        momentum_strategy,
        mean_reversion_strategy
    )
    
    # 导入评估函数
    from evaluation import evaluate_strategy
    
    # 评估基线策略
    baseline_results = {}
    
    # 评估等权重策略
    equal_weight_results = evaluate_strategy(
        strategy_name="等权重策略",
        strategy_function=equal_weight_strategy,
        env=test_env,
        n_stocks=len(ticker_list),
        tech_indicators=tech_indicators
    )
    baseline_results["等权重策略"] = equal_weight_results
    
    # 评估价格反向策略
    price_inverse_results = evaluate_strategy(
        strategy_name="价格反向策略",
        strategy_function=price_inverse_strategy,
        env=test_env,
        n_stocks=len(ticker_list),
        tech_indicators=tech_indicators
    )
    baseline_results["价格反向策略"] = price_inverse_results
    
    # 将RL代理结果添加到比较中
    all_results = {**baseline_results, rl_results["name"]: rl_results}
    
    # 计算基准收益
    benchmark_returns = calculate_benchmark_returns(test_data, ticker_list)
    
    # 创建性能指标表格
    metrics_table = create_metrics_table(all_results)
    print("\n策略性能指标对比:")
    print(metrics_table)
    
    # 保存性能指标表格
    metrics_table.to_csv(os.path.join(RESULTS_DIR, 'comparison_metrics.csv'))
    
    # 绘制投资组合价值对比图
    plot_portfolio_values(all_results, save_dir=RESULTS_DIR, show_plot=False)
    
    # 绘制累积收益率对比图
    plot_returns_comparison(all_results, save_dir=RESULTS_DIR, show_plot=False)
    
    # 绘制与基准比较图
    plot_benchmark_comparison(all_results, benchmark_returns, save_dir=RESULTS_DIR, show_plot=False)
    
    return all_results, benchmark_returns, metrics_table

def main(use_enhanced_reward=False, agent_name="PPO代理"):
    """主函数
    
    参数:
        use_enhanced_reward (bool): 是否使用增强型奖励函数
        agent_name (str): 策略名称
    """
    print("======== 开始PPO代理训练与评估 ========")
    
    # 加载数据
    train_data, test_data, ticker_list = load_data()
    
    # 定义要使用的技术指标
    tech_indicators = ['sma_5', 'sma_10', 'sma_20', 'rsi_14', 
                     'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr_14']
    
    # 创建环境
    train_env, test_env = create_environments(
        train_data, test_data, ticker_list, tech_indicators, use_enhanced_reward
    )
    
    # 设置日志信息
    reward_type = "增强型奖励" if use_enhanced_reward else "标准奖励"
    print(f"使用 {reward_type} 函数创建环境")
    
    # 创建PPOAgent
    state_dim = train_env.observation_space.shape[0]  # 状态空间维度
    action_dim = train_env.action_space.shape[0]      # 动作空间维度
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        batch_size=64,
        device='auto'
    )
    
    # 训练代理
    training_history = train_agent(
        agent=agent,
        env=train_env,
        n_episodes=50,
        max_steps=len(train_data['date'].unique()),
        save_freq=10
    )
    
    # 保存新模型（用不同文件名）
    model_save_path = os.path.join(MODEL_DIR, f"ppo_agent_{agent_name}.pth")
    agent.save(MODEL_DIR, f"ppo_agent_{agent_name}")
    print(f"新模型已保存至 {model_save_path}")
    
    # 评估代理
    rl_results = evaluate_agent(agent, test_env, ticker_list, test_data)
    rl_results["name"] = agent_name  # 用新策略名
    
    # 与基线策略比较
    all_results, benchmark_returns, metrics_table = compare_with_baselines(
        rl_results, test_env, ticker_list, tech_indicators, test_data
    )
    
    print("\n所有结果已保存到 results/rl_agent 目录")
    print("======== 分析完成 ========")
    
    return agent, training_history, all_results

if __name__ == "__main__":
    # 运行时可指定新策略名，如PPO-BaseReward
    import sys
    agent_name = "PPO代理"
    if len(sys.argv) > 1:
        agent_name = sys.argv[1]
    main(agent_name=agent_name)