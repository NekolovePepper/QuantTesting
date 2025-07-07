import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces



# 定义股票交易环境
class StockTradingEnv(gym.Env):
    """
    股票交易环境，继承自gym.Env
    这个环境实现了股票投资组合分配任务
    """
    
    def __init__(self, df, stock_dim, hmax=10, initial_amount=1000000, 
                 transaction_cost_pct=0.001, reward_scaling=1e-4, 
                 state_space=None, action_space=None, tech_indicator_list=None,
                 day=0, lookback=1):
        """
        初始化环境
        
        参数:
            df (pandas.DataFrame): 股票数据，包含日期、代码、OHLCV等信息
            stock_dim (int): 股票数量
            hmax (int): 每次交易最大股票数
            initial_amount (float): 初始资金
            transaction_cost_pct (float): 交易成本百分比
            reward_scaling (float): 奖励缩放因子
            state_space (gym.Space): 状态空间
            action_space (gym.Space): 动作空间
            tech_indicator_list (list): 技术指标列表
            day (int): 当前交易日
            lookback (int): 回溯天数
        """
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        
        # 获取日期列表
        self.dates = self.df['date'].unique()
        self.data = self.df.set_index(['date', 'tic']).sort_index()
        
        # 定义动作空间 (使用Box)
        # 对于每只股票，动作是一个在0和1之间的值，表示投资比例
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # 计算状态空间维度 (此处简化处理)
        if tech_indicator_list is not None:
            # 股票特征: 开盘价、收盘价、最高价、最低价、交易量、持仓比例 + 技术指标
            state_dimension = 6 + len(tech_indicator_list)
        else:
            # 仅使用基本特征
            state_dimension = 6
            
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.stock_dim * state_dimension + 1,), # +1 for cash
            dtype=np.float32
        )
        
        # 初始化状态
        self.reset()
    
    def reset(self):
        """
        重置环境状态
        
        返回:
            observation (numpy.array): 初始状态
        """
        self.terminal = False
        self.day = 0
        
        # 初始化资产组合 (现金和股票)
        self.state = [self.initial_amount] + [0] * self.stock_dim
        self.portfolio_value = self.initial_amount
        
        # 初始化持仓成本
        self.cost = 0
        self.trades = 0
        
        # 初始化回报历史
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        
        # 获取当前日期的状态
        return self._get_observation()
    
    def _get_observation(self):
        """
        根据当前状态构建环境观察
        
        返回:
            observation (numpy.array): 当前状态的观察
        """
        # 获取当前日期
        current_date = self.dates[self.day]

        # 构建观察矩阵
        observation = []
        # 添加当前现金
        observation.append(self.state[0])
        
        # 为每只股票添加特征
        for i in range(self.stock_dim):
            ticker = self.df['tic'].unique()[i]
            
            try:
                # 获取当前股票的数据
                current_tick_data = self.data.loc[(current_date, ticker)]
                
                # 添加基本特征
                observation.append(current_tick_data['open'])
                observation.append(current_tick_data['high'])
                observation.append(current_tick_data['low'])
                observation.append(current_tick_data['close'])
                observation.append(current_tick_data['volume'])
                observation.append(self.state[i+1])  # 当前持仓数量
                
                # 添加技术指标
                if self.tech_indicator_list is not None:
                    for tech in self.tech_indicator_list:
                        observation.append(current_tick_data[tech])
            except KeyError:
                # 处理数据缺失情况
                print(f"警告: 日期 {current_date} 股票 {ticker} 的数据不存在")
                # 添加默认值
                observation.extend([0] * (6 + (len(self.tech_indicator_list) if self.tech_indicator_list else 0)))
        
        return np.array(observation, dtype=np.float32)
    
    def step(self, actions):
        """
        执行一步交易动作
        
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
        
        # 规范化动作（使权重总和为1）
        actions = actions / np.sum(actions) if np.sum(actions) > 0 else actions
        
        # 对每个股票执行交易
        for i in range(self.stock_dim):
            ticker = self.df['tic'].unique()[i]
            current_tick_data = self.data.loc[(current_date, ticker)]
            
            # 计算目标持仓价值
            target_amount = actions[i] * current_value
            
            # 计算目标股票数量
            target_shares = target_amount / current_tick_data['close']
            
            # 当前持仓
            current_shares = self.state[i+1]
            
            # 计算需要买卖的股票数量
            shares_diff = target_shares - current_shares
            
            # 计算交易成本
            transaction_cost = abs(shares_diff) * current_tick_data['close'] * self.transaction_cost_pct
            
            # 更新现金
            self.state[0] -= shares_diff * current_tick_data['close'] + transaction_cost
            
            # 更新持仓
            self.state[i+1] = target_shares
            
            # 更新成本和交易次数
            self.cost += transaction_cost
            if shares_diff != 0:
                self.trades += 1
        
        # 更新到下一天
        self.day += 1
        
        # 计算新的投资组合价值
        new_value = self.state[0]
        for i in range(self.stock_dim):
            ticker = self.df['tic'].unique()[i]
            next_tick_data = self.data.loc[(next_date, ticker)]
            new_value += self.state[i+1] * next_tick_data['close']
        
        # 计算回报 (投资组合价值变化率)
        reward = (new_value - current_value) / current_value
        
        # 应用奖励缩放
        scaled_reward = reward * self.reward_scaling
        
        # 记录资产和回报
        self.portfolio_value = new_value
        self.asset_memory.append(new_value)
        self.rewards_memory.append(reward)
        
        # 获取新观察
        observation = self._get_observation()
        
        return observation, scaled_reward, False, {"portfolio_value": new_value, "reward": reward}
    
    def render(self, mode='human'):
        """
        渲染环境状态
        """
        if mode == 'human':
            print(f"Day: {self.day}, Portfolio Value: {self.portfolio_value}")
    
    def get_final_portfolio_value(self):
        """获取最终投资组合价值"""
        return self.portfolio_value
    
    def get_portfolio_history(self):
        """获取投资组合价值历史"""
        return self.asset_memory
    
    def get_reward_history(self):
        """获取回报历史"""
        return self.rewards_memory



# 定义一个简单的等权重策略
def equal_weight_strategy(observation, n_stocks):
    """返回等权重投资组合"""
    return np.ones(n_stocks) / n_stocks

# 定义一个价格反向策略 (价格低的股票分配更高权重)
def price_inverse_strategy(observation, n_stocks):
    """根据价格反向分配权重"""
    prices = []
    stock_features_length = 6 + len(tech_indicators)  # 6个基本特征 + 技术指标数量
    
    for i in range(n_stocks):
        # 获取每只股票的收盘价 (状态向量中的第4个特征)
        price_idx = 1 + i * stock_features_length + 3  # 1为现金，然后是每只股票的特征，3是收盘价的索引
        if price_idx < len(observation):
            prices.append(observation[price_idx])
        else:
            prices.append(1.0)  # 默认值
    
    # 价格的倒数作为权重
    inverse_prices = 1.0 / (np.array(prices) + 1e-10)  # 避免除零
    weights = inverse_prices / np.sum(inverse_prices)
    
    return weights

# 定义一个基于动量的策略
def momentum_strategy(observation, n_stocks, env):
    """基于短期动量的策略"""
    weights = np.zeros(n_stocks)
    
    # 获取当前日期索引
    day_idx = env.day
    if day_idx < 10:  # 如果历史数据不足，使用等权重
        return np.ones(n_stocks) / n_stocks
    
    # 计算每只股票的10日收益率
    for i in range(n_stocks):
        ticker = ticker_list[i]
        # 获取当前日期
        current_date = env.dates[day_idx]
        # 获取10天前的日期(如果存在)
        past_day_idx = max(0, day_idx - 10)
        past_date = env.dates[past_day_idx]
        
        try:
            current_price = env.data.loc[(current_date, ticker)]['close']
            past_price = env.data.loc[(past_date, ticker)]['close']
            momentum = current_price / past_price - 1
            weights[i] = max(0, momentum)  # 只考虑正动量
        except:
            weights[i] = 0
    
    # 如果所有权重都为0，使用等权重
    if np.sum(weights) == 0:
        return np.ones(n_stocks) / n_stocks
    
    # 归一化权重
    weights = weights / np.sum(weights)
    
    return weights

# Agent classes for testing and comparison
class RandomAgent:
    """
    随机智能体 - 随机分配投资组合权重
    """
    def __init__(self, stock_dim):
        self.stock_dim = stock_dim
    
    def predict(self, state):
        # 生成随机权重并归一化
        weights = np.random.random(self.stock_dim)
        return weights / weights.sum()


class EqualWeightAgent:
    """
    等权重智能体 - 对每只股票分配相同的权重
    """
    def __init__(self, stock_dim):
        self.stock_dim = stock_dim
    
    def predict(self, state):
        # 为每只股票分配相同权重
        return np.ones(self.stock_dim) / self.stock_dim
