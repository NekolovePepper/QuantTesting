import numpy as np
import pandas as pd
def equal_weight_strategy(observation, n_stocks, tech_indicators=None):
    """
    等权重策略 - 每只股票分配相同的权重
    
    参数:
        observation (numpy.array): 环境观察，包含当前市场状态
        n_stocks (int): 股票数量
        tech_indicators (list, optional): 技术指标列表，此策略不使用但为保持接口一致添加
        
    返回:
        numpy.array: 每只股票的权重分配
    """
    return np.ones(n_stocks) / n_stocks

def price_inverse_strategy(observation, n_stocks, tech_indicators=None):
    """
    价格反向策略 - 价格低的股票获得更高权重
    
    参数:
        observation (numpy.array): 环境观察，包含当前市场状态
        n_stocks (int): 股票数量
        tech_indicators (list): 技术指标列表
        
    返回:
        numpy.array: 每只股票的权重分配
    """
    prices = []
    # 计算每只股票特征的长度
    stock_features_length = 6 + (len(tech_indicators) if tech_indicators else 0)  
    
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
def momentum_strategy(observation: np.ndarray, n_stocks: int, env: 'StockTradingEnv') -> np.ndarray:
    """基于短期动量的策略"""
    weights = np.zeros(n_stocks)
    
    # 获取当前日期索引
    # 'env.day' refers to the current step/day index within the env's current df
    day_idx = env.day 
    
    # 'env.dates' are the unique dates from the env's current df
    if day_idx < 10 or len(env.dates) <= 10 : # 如果历史数据不足 (less than 10 previous days in this window)
        # print(f"Momentum: Not enough history (day_idx: {day_idx}, total_days_in_window: {len(env.dates)}), using equal weight.")
        if n_stocks == 0: return np.array([])
        return np.ones(n_stocks) / n_stocks
    
    # 计算每只股票的10日收益率
    for i in range(n_stocks):
        # MODIFICATION: Use the global ticker list stored in the environment
        ticker = env.global_ticker_list[i] 
        
        current_date = env.dates[day_idx]
        # Get the date 10 days ago *within the current window's dates*
        # Ensure past_day_idx is valid for env.dates
        past_day_idx = day_idx - 10 # env.dates[day_idx - 10]
        
        try:
            # env.data is the indexed version of env.df (the current window's data)
            current_price_series = env.data.loc[(current_date, ticker)]
            past_price_series = env.data.loc[(env.dates[past_day_idx], ticker)]

            # Ensure 'close' exists and extract the scalar price
            if 'close' in current_price_series and 'close' in past_price_series:
                current_price = current_price_series['close']
                past_price = past_price_series['close']

                if pd.notna(current_price) and pd.notna(past_price) and past_price > 1e-6:
                    momentum = current_price / past_price - 1
                    weights[i] = max(0, momentum)  # 只考虑正动量
                else:
                    weights[i] = 0 # Price data is NaN or past_price is zero
            else:
                weights[i] = 0 # 'close' column missing for some reason
                
        except KeyError: # If data for the ticker on current_date or past_date is not in env.data
            # print(f"Momentum: Data not found for {ticker} on {current_date} or {env.dates[past_day_idx]}. Weight set to 0.")
            weights[i] = 0
    
    if np.sum(weights) < 1e-6: # If all weights are very close to 0
        # print("Momentum: All calculated momentums are zero or negative, using equal weight.")
        if n_stocks == 0: return np.array([])
        return np.ones(n_stocks) / n_stocks
    
    return weights / np.sum(weights) # 归一化权重
def mean_reversion_strategy(observation, n_stocks, env, tech_indicators=None):
    """
    均值回归策略 - 基于价格偏离移动平均线的程度分配权重
    
    参数:
        observation (numpy.array): 环境观察，包含当前市场状态
        n_stocks (int): 股票数量
        env: 交易环境对象，用于访问数据
        tech_indicators (list): 技术指标列表
        
    返回:
        numpy.array: 每只股票的权重分配
    """
    weights = np.zeros(n_stocks)
    
    if not tech_indicators or 'sma_20' not in tech_indicators:
        # 如果没有提供技术指标或没有20日均线，使用等权重
        return np.ones(n_stocks) / n_stocks
    
    # 确定sma_20在技术指标列表中的位置
    sma_index = tech_indicators.index('sma_20') if 'sma_20' in tech_indicators else -1
    
    if sma_index == -1:
        return np.ones(n_stocks) / n_stocks
        
    # 计算每只股票的状态特征长度
    stock_features_length = 6 + len(tech_indicators)
    
    for i in range(n_stocks):
        # 获取收盘价和20日均线
        price_idx = 1 + i * stock_features_length + 3  # 收盘价索引
        sma20_idx = 1 + i * stock_features_length + 6 + sma_index  # 20日均线索引
        
        if price_idx < len(observation) and sma20_idx < len(observation):
            price = observation[price_idx]
            sma20 = observation[sma20_idx]
            
            if sma20 > 0:
                # 计算价格相对于均线的偏离程度
                deviation = (sma20 - price) / sma20
                # 偏离越大，权重越高 (价格低于均线，预期会上涨)
                weights[i] = max(0, deviation)
            else:
                weights[i] = 0
        else:
            weights[i] = 1.0 / n_stocks  # 默认权重
    
    # 如果所有权重都为0，使用等权重
    if np.sum(weights) == 0:
        return np.ones(n_stocks) / n_stocks
    
    # 归一化权重
    weights = weights / np.sum(weights)
    
    return weights

def volume_weighted_strategy(observation, n_stocks, tech_indicators=None):
    """
    成交量加权策略 - 根据成交量分配权重
    
    参数:
        observation (numpy.array): 环境观察，包含当前市场状态
        n_stocks (int): 股票数量
        tech_indicators (list): 技术指标列表
        
    返回:
        numpy.array: 每只股票的权重分配
    """
    volumes = []
    # 计算每只股票特征的长度
    stock_features_length = 6 + (len(tech_indicators) if tech_indicators else 0)
    
    for i in range(n_stocks):
        # 获取每只股票的成交量 (状态向量中的第5个特征)
        volume_idx = 1 + i * stock_features_length + 4  # 1为现金，然后是每只股票的特征，4是成交量的索引
        if volume_idx < len(observation):
            volumes.append(observation[volume_idx])
        else:
            volumes.append(1.0)  # 默认值
    
    # 成交量作为权重
    volumes_array = np.array(volumes)
    weights = volumes_array / (np.sum(volumes_array) + 1e-10)  # 避免除零
    
    return weights