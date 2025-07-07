import numpy as np
import pandas as pd
import os

def calculate_metrics(portfolio_values):
    """
    计算投资组合的性能指标
    
    参数:
        portfolio_values (list): 投资组合价值历史
        
    返回:
        dict: 包含各种性能指标的字典
    """
    # 转换为numpy数组
    values = np.array(portfolio_values)
    # 计算累积收益率
    returns = values / values[0] - 1
    cumulative_return = returns[-1]
    
    # 计算日收益率
    daily_returns = np.diff(values) / values[:-1]
    
    # 计算夏普比率 (假设年化252个交易日)
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    # 计算最大回撤
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = drawdown.max()
    
    # 计算年化收益率
    n_days = len(values)
    annual_return = (values[-1] / values[0]) ** (252 / n_days) - 1
    
    # 计算索提诺比率 (用下行风险替代标准差)
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.mean(daily_returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
    
    # 计算收益波动率
    volatility = np.std(daily_returns) * np.sqrt(252)
    
    # 计算胜率 (正收益天数比例)
    win_rate = np.sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
    
    return {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "win_rate": win_rate
    }

# In src/evaluation/evaluation.py
import pandas as pd
import numpy as np

def calculate_benchmark_returns(test_df: pd.DataFrame, ticker_list: list) -> pd.Series:
    """
    Calculates the daily returns of an equally weighted "buy and hold" portfolio
    for the given tickers over the test_df period.

    Args:
        test_df (pd.DataFrame): The test dataset, filtered for the specific window.
                                Must contain 'date', 'tic', and 'close' columns.
        ticker_list (list): List of ticker symbols to include in the benchmark.

    Returns:
        pd.Series: Daily returns of the benchmark portfolio, indexed by date.
                   Returns an empty Series if data is insufficient.
    """
    if test_df.empty or not ticker_list:
        return pd.Series(dtype=float)

    # Ensure 'date' is datetime
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # Pivot the table to get close prices with dates as index and tickers as columns
    # Only consider tickers that are actually in the current test_df and in the ticker_list
    relevant_tickers = [tic for tic in ticker_list if tic in test_df['tic'].unique()]
    if not relevant_tickers:
        return pd.Series(dtype=float)

    close_prices_pivot = test_df[test_df['tic'].isin(relevant_tickers)].pivot_table(
        index='date', 
        columns='tic', 
        values='close'
    )
    
    # Forward fill any missing values within a ticker's series, then backfill
    # This handles days where a specific stock might not have a price, but others do.
    close_prices_pivot = close_prices_pivot.ffill().bfill()

    # Drop any columns (tickers) that are still all NaN after ffill/bfill
    # (meaning they had no data at all in this window)
    close_prices_pivot = close_prices_pivot.dropna(axis=1, how='all')

    if close_prices_pivot.empty or close_prices_pivot.shape[1] == 0:
         print("警告 (calculate_benchmark_returns): 数据透视表为空或没有有效的股票列。")
         return pd.Series(dtype=float)

    # Calculate daily returns for each stock
    # pct_change() handles NaNs correctly by propagating them for the first day of change.
    daily_stock_returns = close_prices_pivot.pct_change()

    # Calculate the mean daily return across stocks (equally weighted portfolio)
    # This will ignore NaNs in the mean calculation for each row (day) by default.
    # If on a particular day, some stocks have NaN returns (e.g., first day), they are excluded from the mean for that day.
    benchmark_daily_returns = daily_stock_returns.mean(axis=1)
    
    # The first day's return will be NaN due to pct_change(), fill it with 0.
    benchmark_daily_returns = benchmark_daily_returns.fillna(0)
    
    # `benchmark_daily_returns` is now a pd.Series of the benchmark's daily returns.
    # Your evaluation functions might then calculate cumulative returns from this.
    # Or, if you need cumulative:
    # benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod() - 1
    # return benchmark_cumulative_returns # If you want cumulative
    
    return benchmark_daily_returns # Return daily returns

def evaluate_strategy(strategy_name, strategy_function, env, n_stocks, tech_indicators=None, verbose=True):
    """
    评估单个投资策略
    
    参数:
        strategy_name (str): 策略名称
        strategy_function (function): 策略函数
        env: 交易环境对象
        n_stocks (int): 股票数量
        tech_indicators (list): 技术指标列表
        verbose (bool): 是否打印评估结果
        
    返回:
        dict: 包含评估结果的字典
    """
    # 重置环境
    env.reset()
    done = False
    total_reward = 0
    portfolio_values = []
    
    # 执行策略
    while not done:
        observation = env._get_observation()
        
        # 根据策略类型处理不同参数
        if strategy_function.__name__ == 'momentum_strategy':
            action = strategy_function(observation, n_stocks, env)
        elif strategy_function.__name__ == 'mean_reversion_strategy':
            action = strategy_function(observation, n_stocks, env, tech_indicators)
        else:
            action = strategy_function(observation, n_stocks, tech_indicators)
            
        observation, reward, done, info = env.step(action)
        total_reward += reward
        portfolio_values.append(env.portfolio_value)
    
    # 计算性能指标
    metrics = calculate_metrics(portfolio_values)
    
    if verbose:
        print(f"\n{strategy_name} 评估结果:")
        print(f"累计回报率: {metrics['cumulative_return']:.4f}")
        print(f"年化收益率: {metrics['annual_return']:.4f}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"索提诺比率: {metrics['sortino_ratio']:.4f}")
        print(f"最大回撤: {metrics['max_drawdown']:.4f}")
        print(f"波动率: {metrics['volatility']:.4f}")
        print(f"胜率: {metrics['win_rate']:.4f}")
    
    # 返回结果字典
    return {
        "name": strategy_name,
        "metrics": metrics,
        "portfolio_values": portfolio_values,
        "total_reward": total_reward
    }

def create_metrics_table(strategy_results):
    """
    创建性能指标表格
    
    参数:
        strategy_results (dict): 策略评估结果字典
        
    返回:
        pandas.DataFrame: 性能指标表格
    """
    metrics_list = []
    
    for name, results in strategy_results.items():
        metrics = results['metrics']
        metrics['Strategy'] = name
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.set_index('Strategy')
    
    # 重命名列以便更易读
    metrics_df = metrics_df.rename(columns={
        'cumulative_return': '累计收益率',
        'annual_return': '年化收益率',
        'sharpe_ratio': '夏普比率',
        'sortino_ratio': '索提诺比率',
        'max_drawdown': '最大回撤',
        'volatility': '波动率',
        'win_rate': '胜率'
    })
    
    # 设置显示格式
    pd.options.display.float_format = '{:.4f}'.format
    
    return metrics_df

def update_strategy_metrics(new_result, csv_path):
    """
    更新或新增策略评估结果到csv（按策略名字符串匹配）
    参数：
        new_result (dict): 新的策略评估结果，key为列名
        csv_path (str): csv文件路径
    """
    import pandas as pd
    import os
    # 读取原有csv
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # 如果文件不存在，直接用新结果创建
        df = pd.DataFrame([new_result])
        df.to_csv(csv_path, index=False)
        return
    # 查找是否有同名策略
    match = df['Strategy'].astype(str) == str(new_result['Strategy'])
    if match.any():
        # 有同名，更新该行
        df.loc[match, :] = pd.DataFrame([new_result])
    else:
        # 没有同名，新增一行
        df = pd.concat([df, pd.DataFrame([new_result])], ignore_index=True)
    df.to_csv(csv_path, index=False)

def plot_portfolio_values(portfolio_values, title="Portfolio Value Over Time", save_path=None, show=False):
    """
    绘制投资组合价值曲线

    参数:
        portfolio_values (list or np.array): 投资组合价值序列
        title (str): 图表标题
        save_path (str): 图片保存路径（如为None则不保存）
        show (bool): 是否直接显示图像
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_returns_comparison(returns_dict, title="Returns Comparison", save_path=None, show=False):
    """
    多策略收益率曲线对比绘图

    参数:
        returns_dict (dict): {策略名: 收益率序列}
        title (str): 图表标题
        save_path (str): 图片保存路径
        show (bool): 是否直接显示
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    for name, returns in returns_dict.items():
        plt.plot(returns, label=name)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_benchmark_comparison(benchmark_dict, title="Benchmark Comparison", save_path=None, show=False):
    """
    多策略与基准收益/价值曲线对比绘图

    参数:
        benchmark_dict (dict): {策略名: 序列}
        title (str): 图表标题
        save_path (str): 图片保存路径
        show (bool): 是否直接显示
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    if benchmark_returns is not None and not benchmark_returns.empty:
        cumulative_benchmark_returns = (1 + benchmark_returns).cumprod()
        # Adjust portfolio values to start from 1 for direct comparison if they are not already
        # Example: normalized_portfolio_values = results['portfolio_values'] / results['portfolio_values'][0]
        plt.plot(cumulative_benchmark_returns.index, cumulative_benchmark_returns, label='Benchmark (Buy & Hold Avg)', linestyle='--')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value/Return")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()