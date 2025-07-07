

import os
import numpy as np
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    """
    股票交易环境，继承自gym.Env
    这个环境实现了股票投资组合分配任务
    """
    
    def __init__(self, df, stock_dim, hmax=10, initial_amount=1000000, 
                 global_ticker_list=None,
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
        for i in range(self.stock_dim):  # 调试断点
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
        total_investment = current_value
        # min(current_value, available_cash)  # 只能使用可用资金进行重新分配
        
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
        
        # 计算奖励，引入多个因素
        # 1. 基础奖励：每日收益率
        base_reward = daily_return
        
        # 2. 风险惩罚：计算短期波动率惩罚（如果有足够的历史数据）
        risk_penalty = 0
        if len(self.asset_memory) >= 5:  # 至少需要5天的数据来计算波动率
            # 计算最近5天的收益率
            recent_returns = []
            for i in range(max(0, len(self.asset_memory)-5), len(self.asset_memory)):
                if i > 0:  # 跳过第一天
                    ret = (self.asset_memory[i] / self.asset_memory[i-1]) - 1
                    recent_returns.append(ret)
            
            # 计算波动率
            if len(recent_returns) > 1:
                volatility = np.std(recent_returns)
                # 波动率惩罚：波动率越高，惩罚越大
                risk_penalty = volatility * 0.5  # 可调整系数
        
        # 3. 计算投资组合多样性奖励
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
        diversity_reward = normalized_entropy * 0.2  # 可调整系数
        
        # 4. 交易成本惩罚
        transaction_penalty = (self.cost / current_value) * 0.1 if current_value > 0 else 0
        
        # 5. 最大回撤惩罚
        drawdown_penalty = 0
        if len(self.asset_memory) > 1:
            historical_max = max(self.asset_memory)
            if historical_max > 0:
                current_drawdown = (historical_max - new_value) / historical_max
                drawdown_penalty = current_drawdown * 0.3  # 可调整系数
        
        # 综合计算奖励
        reward = base_reward - risk_penalty + diversity_reward - transaction_penalty - drawdown_penalty
        
        # 限制奖励在合理范围内
        reward = np.clip(reward, -1.0, 1.0)
        
        # 应用奖励缩放
        scaled_reward = reward * self.reward_scaling
        
        # 记录资产和回报
        self.portfolio_value = new_value
        self.asset_memory.append(new_value)
        self.rewards_memory.append(reward)
        
        # 获取新观察
        observation = self._get_observation()
        
        return observation, scaled_reward, False, {
            "portfolio_value": new_value, 
            "reward": reward,
            "base_reward": base_reward,
            "risk_penalty": risk_penalty,
            "diversity_reward": diversity_reward,
            "transaction_penalty": transaction_penalty,
            "drawdown_penalty": drawdown_penalty
        }
    
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
    '''
import gym # 请确保 gym 已导入
from gym import spaces # 请确保 spaces 已导入
import numpy as np # 请确保 numpy 已导入
import pandas as pd # 请确保 pandas 已导入

# 假设 EnhancedRewardStockTradingEnv 如果存在，也需要做类似修改或继承已修改的 StockTradingEnv
# from src.core.enhanced_env import EnhancedRewardStockTradingEnv # 如果您有这个类

class StockTradingEnv(gym.Env):
    """
    股票交易环境，继承自gym.Env
    这个环境实现了股票投资组合分配任务
    """
    
    def __init__(self, df: pd.DataFrame, 
                 stock_dim: int, 
                 global_ticker_list: list, # 新增参数：全局股票代码列表
                 hmax: int = 10, 
                 initial_amount: float = 1000000, 
                 transaction_cost_pct: float = 0.001, 
                 reward_scaling: float = 1e-4, 
                 tech_indicator_list: list | None = None,
                 day: int = 0): # 移除了 state_space, action_space, lookback (因为它们在原代码中未被有效使用或在内部定义)
        """
        初始化环境
        
        参数:
            df (pandas.DataFrame): 当前窗口的股票数据
            stock_dim (int): 股票数量 (基于全局列表)
            global_ticker_list (list): 全局股票代码列表，顺序与 stock_dim 对应
            hmax (int): 每次交易最大股票数
            initial_amount (float): 初始资金
            transaction_cost_pct (float): 交易成本百分比
            reward_scaling (float): 奖励缩放因子
            tech_indicator_list (list, optional): 技术指标列表
            day (int): 当前交易日（相对于传入的df的起始）
        """
        super(StockTradingEnv, self).__init__() # 规范的gym.Env继承初始化
        
        self.day = day
        self.df = df # 当前窗口的数据
        self.stock_dim = stock_dim # 基于全局股票列表的维度
        self.global_ticker_list = global_ticker_list # 存储全局股票列表
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list if tech_indicator_list is not None else [] # 确保是列表
        
        # 获取当前窗口的日期列表
        self.dates = self.df['date'].unique()
        # 使用传入的、可能只是部分股票数据的df来创建索引数据，用于快速查找
        # 注意：如果df为空，这里会出错，调用者应保证df非空
        if self.df.empty:
            # 如果df为空，环境无法正确初始化。这通常意味着上游数据加载存在问题。
            # 为了避免崩溃，我们可以设置一个标记，或者抛出错误。
            # 这里我们允许它继续，但后续步骤可能会因为self.dates为空而出错。
            # 更好的做法是在create_environments中检查并阻止空df传入。
            print("警告 (StockTradingEnv): 传入的DataFrame为空。环境可能无法正常工作。")
            self.data = pd.DataFrame() # 空的DataFrame
        else:
            self.data = self.df.set_index(['date', 'tic']).sort_index()
        
        # 定义动作空间 (使用Box)
        # 对于每只股票，动作是一个在0和1之间的值，表示投资比例
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # 计算状态空间中每个股票的特征数量
        # 股票特征: 开盘价、收盘价、最高价、最低价、交易量、持仓比例 + 技术指标
        self.num_stock_features = 6 + len(self.tech_indicator_list)
            
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(1 + self.stock_dim * self.num_stock_features,), # +1 for cash
            dtype=np.float32
        )
        
        # 初始化状态
        self.reset() # reset会调用_get_observation
    
    def reset(self):
        """
        重置环境状态
        
        返回:
            observation (numpy.array): 初始状态
        """
        self.terminal = False
        self.day = 0 # 重置到当前数据窗口的第0天
        
        # 初始化资产组合 (现金和股票持仓数量)
        # state[0] 是现金，state[1:] 是各股票的持仓数量 (不是价值)
        self.state = [self.initial_amount] + [0.0] * self.stock_dim # 使用浮点数表示持仓
        self.portfolio_value = self.initial_amount
        
        self.cost = 0.0
        self.trades = 0
        
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        
        # 检查self.dates是否为空，如果为空，无法获取观察值
        if len(self.dates) == 0:
            # 返回一个符合观察空间形状的零数组或者抛出错误
            print("错误 (StockTradingEnv.reset): 日期列表为空，无法获取初始观察。")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        return self._get_observation()
    
    def _get_observation(self):
        """
        根据当前状态构建环境观察
        
        返回:
            observation (numpy.array): 当前状态的观察
        """
        # 如果日期列表为空（可能因为传入的df为空），则无法继续
        if not self.dates.size:
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        current_date = self.dates[self.day]
        observation = []
        observation.append(self.state[0]) # 当前现金
        
        # 为每只股票添加特征，使用全局股票列表进行迭代
        for i in range(self.stock_dim):
            ticker = self.global_ticker_list[i] # <--- 改动点：使用全局列表获取股票代码
            
            try:
                # 获取当前股票在当前窗口数据中的数据
                current_tick_data = self.data.loc[(current_date, ticker)]
                
                observation.append(current_tick_data['open'])
                observation.append(current_tick_data['high'])
                observation.append(current_tick_data['low'])
                observation.append(current_tick_data['close'])
                observation.append(current_tick_data['volume'])

                observation.append(self.state[i+1])  # 当前持仓数量 (shares)
                
                for tech in self.tech_indicator_list:
                    observation.append(current_tick_data[tech])
            except KeyError:
                # 如果当前股票在当前日期没有数据（例如，在此窗口中该股票数据缺失）
                # 则为该股票的所有特征填充0，以保持观察向量的维度一致
                # print(f"警告 (_get_observation): 日期 {current_date} 股票 {ticker} 的数据在当前窗口不存在，使用0填充。")
                observation.extend([0.0] * self.num_stock_features) # 使用0.0填充浮点数
        
        return np.array(observation, dtype=np.float32)
    
    def step(self, actions: np.ndarray):
        """
        执行一步交易动作
        """
        # 检查是否已经到达数据窗口的末尾
        self.terminal = self.day >= len(self.dates) - 1 # len(self.dates)-1 是最后一个有效索引
        
        if self.terminal:
            # 如果已经到达终点，计算最后一次的资产组合价值并返回
            # _get_observation 理论上不应在 terminal 后调用来获取 next_state
            # 但如果需要返回最终观察，可以调用，或者返回一个特殊的最终观察
            # 这里的逻辑是返回当前观察，奖励为0，done为True
            
            # 在结束前，计算最终的投资组合价值
            final_portfolio_value = self.state[0] # 现金
            if len(self.dates) > 0 : # 确保 self.dates 非空
                current_date_for_final_value = self.dates[self.day] # 使用当前（最后一天）的价格
                for i in range(self.stock_dim):
                    ticker = self.global_ticker_list[i]
                    try:
                        # 使用当前日期的收盘价计算股票价值
                        tick_data = self.data.loc[(current_date_for_final_value, ticker)]
                        final_portfolio_value += self.state[i+1] * tick_data['close']
                    except KeyError:
                        pass # 如果股票数据缺失，其价值贡献为0
            
            self.portfolio_value = final_portfolio_value
            if self.asset_memory[-1] != self.portfolio_value: # 避免重复添加
                 self.asset_memory.append(self.portfolio_value)

            # print(f"环境结束于第 {self.day} 天。最终资产组合价值: {self.portfolio_value:.2f}")
            # 确保返回的观察符合维度
            final_obs = self._get_observation() if len(self.dates)>0 else np.zeros(self.observation_space.shape, dtype=np.float32)

            return final_obs, 0.0, True, {"portfolio_value": self.portfolio_value, "reward": 0.0, 'terminal': True}

        # 获取当前日期和下一个交易日用于计算
        current_date = self.dates[self.day]
        
        # 计算交易前的投资组合价值 (基于当日收盘价)
        begin_portfolio_value = self.state[0] # 当前现金
        for i in range(self.stock_dim):
            ticker = self.global_ticker_list[i]
            try:
                tick_data = self.data.loc[(current_date, ticker)]
                begin_portfolio_value += self.state[i+1] * tick_data['close']
            except KeyError: # 如果某支股票在当前日期没有数据，则其持仓价值贡献视为0
                pass 
        self.portfolio_value = begin_portfolio_value # 更新一下当前的资产组合价值记录


        # 规范化动作（使目标权重总和为1）
        if np.sum(actions) > 1e-6: # 避免除以非常小的数或零
            actions = actions / np.sum(actions)
        else: # 如果所有动作都接近于0，则不持有任何股票（或保持现状，这里设为全卖出/不买入）
            actions = np.zeros(self.stock_dim, dtype=np.float32)
        
        # 根据动作调整仓位
        # new_state_cash = self.state[0] # 从当前现金开始计算
        # new_state_shares = list(self.state[1:]) # 当前持股
        
        target_allocations_value = actions * self.portfolio_value # 每只股票的目标持有价值

        # 卖出操作：首先计算卖出股票所得现金，并更新持股为0（逻辑简化为先卖后买）
        # 实际上，更真实的模拟是计算与目标的差异。
        # 我们这里采用 FinRL 的标准方法：计算目标持股数，然后买入或卖出差额。
        
        for i in range(self.stock_dim):
            ticker = self.global_ticker_list[i]
            try:
                current_price = self.data.loc[(current_date, ticker)]['close']
                if current_price <= 1e-6: # 价格过低或为0，无法交易
                    target_shares_for_stock_i = 0.0
                else:
                    target_shares_for_stock_i = target_allocations_value[i] / current_price
                    target_shares_for_stock_i = min(target_shares_for_stock_i, self.hmax) # 应用hmax限制单只股票最大持仓（这里hmax解释为股数）
                                                                                        # 如果hmax是价值，则应在target_allocations_value处限制

            except KeyError: # 当日无此股票数据
                target_shares_for_stock_i = 0.0 # 无法交易，目标设为0
                current_price = 0.0

            current_shares_for_stock_i = self.state[i+1]
            shares_to_trade = target_shares_for_stock_i - current_shares_for_stock_i
            
            trade_value = abs(shares_to_trade) * (current_price if current_price > 1e-6 else 0) # 交易额
            transaction_cost = trade_value * self.transaction_cost_pct
            
            self.state[0] -= shares_to_trade * (current_price if current_price > 1e-6 else 0) # 更新现金，买入减现金，卖出加现金
            self.state[0] -= transaction_cost # 扣除交易成本
            self.state[i+1] = target_shares_for_stock_i # 更新持股数量
            
            self.cost += transaction_cost
            if abs(shares_to_trade) > 1e-6 : # 有实际交易发生
                self.trades += 1
        
        # 更新到下一天
        self.day += 1
        next_date = self.dates[self.day] # 获取下一天的日期
        
        # 计算下一天开始时的投资组合价值 (基于下一天的收盘价)
        end_portfolio_value = self.state[0] # 更新后的现金
        for i in range(self.stock_dim):
            ticker = self.global_ticker_list[i]
            try:
                tick_data = self.data.loc[(next_date, ticker)] # 使用下一天的价格数据
                end_portfolio_value += self.state[i+1] * tick_data['close']
            except KeyError: # 如果下一天某支股票数据缺失
                pass

        # 计算回报 (是这一步操作后，持有到下一天资产组合价值的变化)
        # Reward is based on the change in portfolio value from before taking action (at current_date prices)
        # to after holding the new portfolio until next_date (at next_date prices)
        reward_value = end_portfolio_value - self.portfolio_value # self.portfolio_value 是交易前的价值
        
        # 应用奖励缩放
        scaled_reward = reward_value * self.reward_scaling
        
        # 更新资产和回报记录
        self.portfolio_value = end_portfolio_value # 更新资产组合价值为下一天开始时的价值
        self.asset_memory.append(self.portfolio_value)
        self.rewards_memory.append(reward_value) # 记录未缩放的原始奖励值
        
        observation = self._get_observation() # 获取下一天的观察

        return observation, scaled_reward, self.terminal, {"portfolio_value": self.portfolio_value, "reward": reward_value}
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Day: {self.day}, Portfolio Value: {self.portfolio_value:.2f}")
    
    def get_final_portfolio_value(self):
        return self.portfolio_value
    
    def get_portfolio_history(self):
        return self.asset_memory
    
    def get_reward_history(self):
        return self.rewards_memory
'''