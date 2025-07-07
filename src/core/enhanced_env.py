import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.core.env import StockTradingEnv
from src.baseline.baseline_agent import RandomAgent, EqualWeightAgent
import os

# 创建修改后的环境类，使用新的奖励函数
class EnhancedRewardStockTradingEnv(StockTradingEnv):
    """
    增强型奖励函数的股票交易环境
    """
    
    def step(self, actions):
        """
        执行一步交易动作，并使用增强型奖励函数
        
        参数:
            actions (numpy.array): 交易动作，表示每只股票的目标权重
            
        返回:
            observation (numpy.array): 新的观察
            reward (float): 回报
            done (bool): 是否结束
            info (dict): 附加信息
        """
        self.terminal = self.day >= len(self.dates) - 1
        
        if self.terminal:
            # 如果已经到达终点，返回最后状态
            return self._get_observation(), 0, True, {'terminal': True}
        
        # 获取当前日期和下一个日期
        current_date = self.dates[self.day]
        next_date = self.dates[self.day + 1]
        
        # 计算当前投资组合价值
        current_value = self.state[0]
        for i in range(self.stock_dim):
            ticker = self.df['tic'].unique()[i]
            current_tick_data = self.data.loc[(current_date, ticker)]
            current_value += self.state[i+1] * current_tick_data['close']
        
        # 确保actions是一个向量而不是标量
        if np.isscalar(actions):
            actions = np.array([actions] * self.stock_dim)
        
        # 确保所有动作值为非负数
        actions = np.clip(actions, 0, None)
        
        # 规范化动作（使权重总和为1）
        action_sum = np.sum(actions)
        if action_sum > 1e-6:  # 避免除以接近零的数
            actions = actions / action_sum
        else:
            # 如果所有动作都接近0，使用均匀分布
            actions = np.ones_like(actions) / self.stock_dim
        
        # 对每个股票执行交易
        available_cash = max(0, self.state[0])  # 确保现金不为负
        
        # 计划交易前先计算总投资额
        total_investment = min(current_value, available_cash)  # 只能使用可用资金进行重新分配
        
        # 记录交易前的权重分布（用于计算投资组合多样性）
        pre_trade_weights = np.zeros(self.stock_dim)
        for i in range(self.stock_dim):
            ticker = self.df['tic'].unique()[i]
            current_tick_data = self.data.loc[(current_date, ticker)]
            stock_value = self.state[i+1] * current_tick_data['close']
            pre_trade_weights[i] = stock_value / current_value if current_value > 0 else 0
            
        # 对每个股票执行交易
        for i in range(self.stock_dim):
            ticker = self.df['tic'].unique()[i]
            current_tick_data = self.data.loc[(current_date, ticker)]
            
            # 计算目标持仓价值
            target_amount = actions[i] * total_investment
            
            # 计算目标股票数量
            current_price = max(0.01, current_tick_data['close'])  # 确保价格大于0
            target_shares = target_amount / current_price
            
            # 当前持仓
            current_shares = self.state[i+1]
            
            # 计算需要买卖的股票数量
            shares_diff = target_shares - current_shares
            
            # 计算交易成本
            transaction_cost = abs(shares_diff) * current_price * self.transaction_cost_pct
            
            # 更新现金 (确保不会为负)
            cash_change = -(shares_diff * current_price + transaction_cost)
            if self.state[0] + cash_change < 0:
                # 如果现金不足，调整购买量
                max_buy_cash = max(0, self.state[0] - transaction_cost)
                max_shares = max_buy_cash / current_price
                if shares_diff > 0:
                    shares_diff = max_shares
                    transaction_cost = abs(shares_diff) * current_price * self.transaction_cost_pct
                    cash_change = -(shares_diff * current_price + transaction_cost)
            
            # 更新现金和持仓
            self.state[0] += cash_change
            self.state[i+1] = current_shares + shares_diff
            
            # 更新成本和交易次数
            self.cost += transaction_cost
            if abs(shares_diff) > 1e-6:  # 仅当有实际交易时增加交易计数
                self.trades += 1
        
        # 更新到下一天
        self.day += 1
        
        # 计算新的投资组合价值
        new_value = max(0.01, self.state[0])  # 确保投资组合价值大于0
        for i in range(self.stock_dim):
            ticker = self.df['tic'].unique()[i]
            next_tick_data = self.data.loc[(next_date, ticker)]
            stock_value = self.state[i+1] * next_tick_data['close']
            new_value += stock_value
        
        # 防止新值为0导致的计算错误
        current_value = max(0.01, current_value)
        
        # 计算回报 (投资组合价值变化率)
        daily_return = (new_value / current_value) - 1
        
        # === 新的奖励函数计算 ===
        
        # 1. 基础奖励：每日收益率 (权重增加到0.5)
        base_reward = daily_return * 0.5
        
        # 2. 风险惩罚：计算夏普比率而不仅仅是波动率
        risk_reward = 0
        if len(self.asset_memory) >= 10:  # 至少需要10天的数据来计算夏普比率
            # 计算最近10天的收益率
            recent_returns = []
            for i in range(max(0, len(self.asset_memory)-10), len(self.asset_memory)):
                if i > 0:  # 跳过第一天
                    ret = (self.asset_memory[i] / self.asset_memory[i-1]) - 1
                    recent_returns.append(ret)
            
            if len(recent_returns) > 1:
                # 计算平均收益率
                mean_return = np.mean(recent_returns)
                
                # 计算波动率
                volatility = np.std(recent_returns)
                
                # 计算日度夏普比率 (假设无风险利率为0，简化计算)
                if volatility > 0:
                    sharpe_ratio = mean_return / volatility
                    # 夏普比率奖励：夏普比率越高，奖励越大
                    risk_reward = np.clip(sharpe_ratio * 0.3, -0.3, 0.3)  # 限制在合理范围内
                else:
                    # 如果波动率为0，给出一个小的正奖励
                    risk_reward = 0.1
        
        # 3. 计算投资组合多样性奖励 (权重保持为0.2)
        diversity_reward = 0
        # 计算投资后的权重分布
        post_trade_weights = np.zeros(self.stock_dim)
        for i in range(self.stock_dim):
            ticker = self.df['tic'].unique()[i]
            next_tick_data = self.data.loc[(next_date, ticker)]
            stock_value = self.state[i+1] * next_tick_data['close']
            post_trade_weights[i] = stock_value / new_value if new_value > 0 else 0
        
        # 计算熵作为多样性指标 (熵越大，多样性越好)
        epsilon = 1e-15  # 防止log(0)
        entropy = -np.sum(post_trade_weights * np.log(post_trade_weights + epsilon))
        max_entropy = np.log(self.stock_dim)  # 均匀分布的最大熵
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # 多样性奖励：归一化的熵值
        diversity_reward = normalized_entropy * 0.2  # 保持系数不变
        
        # 4. 交易成本惩罚 (降低权重到0.05)
        transaction_penalty = (self.cost / current_value) * 0.05 if current_value > 0 else 0
        
        # 5. 最大回撤惩罚 (增加权重到0.4)
        drawdown_penalty = 0
        if len(self.asset_memory) > 1:
            historical_max = max(self.asset_memory)
            if historical_max > 0:
                current_drawdown = (historical_max - new_value) / historical_max
                drawdown_penalty = current_drawdown * 0.4  # 可调整系数
        
        # 6. 新增：长期与短期回报平衡
        long_term_reward = 0
        if len(self.asset_memory) >= 20:  # 需要至少20天的历史数据
            # 计算长期回报率 (20天)
            long_term_return = (new_value / self.asset_memory[max(0, len(self.asset_memory)-20)]) - 1
            long_term_reward = long_term_return * 0.2  # 长期回报的权重为0.2
            
            # 限制在合理范围内
            long_term_reward = np.clip(long_term_reward, -0.2, 0.2)
        
        # 综合计算奖励
        reward = base_reward + risk_reward + diversity_reward - transaction_penalty - drawdown_penalty + long_term_reward
        
        # 限制奖励在合理范围内
        reward = np.clip(reward, -1.0, 1.0)
        
        # 应用奖励缩放
        scaled_reward = reward * self.reward_scaling
        
        # 记录资产和回报
        self.portfolio_value = new_value
        self.asset_memory.append(new_value)
        self.rewards_memory.append(scaled_reward)
        
        # 获取新的观察
        observation = self._get_observation()
        
        # 返回结果
        info = {
            'portfolio_value': new_value,
            'daily_return': daily_return,
            'base_reward': base_reward,
            'risk_reward': risk_reward,
            'diversity_reward': diversity_reward,
            'transaction_penalty': transaction_penalty,
            'drawdown_penalty': drawdown_penalty,
            'long_term_reward': long_term_reward,
            'total_reward': reward,
            'scaled_reward': scaled_reward
        }
        
        return observation, scaled_reward, False, info


def test_reward_functions(data_path='data/stock_data_with_features.csv', 
                         episodes=5, max_steps=100):
    """
    测试并比较原始奖励函数和增强奖励函数
    
    参数:
        data_path: 股票数据路径
        episodes: 测试回合数
        max_steps: 每回合最大步数
    """
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 确保数据包含必要的列
    required_columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in required_columns), "数据缺少必要的列"
    
    # 获取股票数量
    stock_dim = len(df['tic'].unique())
    
    # 初始化原始环境和增强环境
    original_env = StockTradingEnv(df=df, stock_dim=stock_dim)
    enhanced_env = EnhancedRewardStockTradingEnv(df=df, stock_dim=stock_dim)
    
    # 初始化测试智能体 (我们使用均等权重智能体进行测试)
    agent = EqualWeightAgent(stock_dim=stock_dim)
    
    # 用于记录结果的变量
    original_rewards = []
    enhanced_rewards = []
    original_portfolio_values = []
    enhanced_portfolio_values = []
    
    # 测试原始环境
    print("测试原始奖励函数...")
    for episode in range(episodes):
        state = original_env.reset()
        done = False
        episode_rewards = []
        step_count = 0
        
        while not done and step_count < max_steps:
            action = agent.predict(state)
            next_state, reward, done, info = original_env.step(action)
            episode_rewards.append(reward)
            state = next_state
            step_count += 1
        
        original_rewards.append(np.sum(episode_rewards))
        original_portfolio_values.append(original_env.asset_memory)
        print(f"回合 {episode+1}/{episodes} 完成，总奖励: {np.sum(episode_rewards):.4f}, 最终投资组合价值: {original_env.portfolio_value:.2f}")
    
    # 测试增强环境
    print("\n测试增强奖励函数...")
    for episode in range(episodes):
        state = enhanced_env.reset()
        done = False
        episode_rewards = []
        step_count = 0
        
        # 记录每个奖励组件的贡献
        reward_components = {
            'base_reward': [],
            'risk_reward': [],
            'diversity_reward': [],
            'transaction_penalty': [],
            'drawdown_penalty': [],
            'long_term_reward': []
        }
        
        while not done and step_count < max_steps:
            action = agent.predict(state)
            next_state, reward, done, info = enhanced_env.step(action)
            episode_rewards.append(reward)
            
            # 记录奖励组件
            for key in reward_components:
                if key in info:
                    reward_components[key].append(info[key])
            
            state = next_state
            step_count += 1
        
        enhanced_rewards.append(np.sum(episode_rewards))
        enhanced_portfolio_values.append(enhanced_env.asset_memory)
        print(f"回合 {episode+1}/{episodes} 完成，总奖励: {np.sum(episode_rewards):.4f}, 最终投资组合价值: {enhanced_env.portfolio_value:.2f}")
        
        # 打印奖励组件统计
        print("奖励组件统计:")
        for key, values in reward_components.items():
            if values:
                print(f"  {key}: 平均值={np.mean(values):.4f}, 总和={np.sum(values):.4f}")
    
    # 比较结果
    print("\n结果比较:")
    print(f"原始奖励函数 - 平均累积奖励: {np.mean(original_rewards):.4f}")
    print(f"增强奖励函数 - 平均累积奖励: {np.mean(enhanced_rewards):.4f}")
    
    # 创建结果目录
    results_dir = "results/reward_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    # 绘制奖励比较图
    plt.figure(figsize=(12, 6))
    plt.plot(range(episodes), original_rewards, 'b-', label='原始奖励函数')
    plt.plot(range(episodes), enhanced_rewards, 'r-', label='增强奖励函数')
    plt.xlabel('episodes')
    plt.ylabel('accumulated rewards')
    plt.title('reward comparison between original and enhanced reward functions')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/reward_comparison.png")
    
    # 绘制投资组合价值比较图 (使用最后一个回合的数据)
    plt.figure(figsize=(12, 6))
    plt.plot(original_portfolio_values[-1], 'b-', label='原始奖励函数')
    plt.plot(enhanced_portfolio_values[-1], 'r-', label='增强奖励函数')
    plt.xlabel('trading days')
    plt.ylabel('portfolio value')
    plt.title('Portfolio Value Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/portfolio_value_comparison.png")
    
    # 显示结果
    print(f"\n结果已保存到 {results_dir} 目录")


if __name__ == "__main__":
    # 执行测试
    test_reward_functions()