import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Added for potentially handling portfolio_values as Series more robustly

def _format_suffix_for_title(suffix: str) -> str:
    """Helper function to format the suffix for display in plot titles."""
    if not suffix:
        return ""
    # Example: _testyear_2019 -> (Test Year 2019)
    formatted = suffix.replace("_", " ").strip().title()
    if formatted:
        return f" ({formatted})"
    return ""

def plot_portfolio_values(strategy_results: dict, save_dir: str | None = None, show_plot: bool = True, suffix: str = ""):
    """
    Plots the portfolio values of all strategies.

    Args:
        strategy_results (dict): Dictionary of strategy results,
                                 each containing 'portfolio_values'.
        save_dir (str, optional): Directory to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        suffix (str, optional): Suffix to append to the plot filename and title. Defaults to "".
    """
    plt.figure(figsize=(12, 6))
    for name, results in strategy_results.items():
        if 'portfolio_values' not in results or results['portfolio_values'] is None:
            print(f"警告 (plot_portfolio_values): 策略 '{name}' 缺少 'portfolio_values'。")
            continue
        portfolio_values = results['portfolio_values']
        if isinstance(portfolio_values, (list, np.ndarray)) and len(portfolio_values) == 0:
            print(f"警告 (plot_portfolio_values): 策略 '{name}' 的 'portfolio_values' 为空。")
            continue
        if not isinstance(portfolio_values, pd.Series): # Convert to Series for easier handling
            portfolio_values = pd.Series(portfolio_values)

        plt.plot(portfolio_values.index, portfolio_values, label=name) # Use index if Series

    title_suffix = _format_suffix_for_title(suffix)
    plt.title(f'Portfolio Value Comparison{title_suffix}')
    plt.xlabel('Trading Day / Time')
    plt.ylabel('Portfolio Value')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'portfolio_value_comparison{suffix}.png'
        
        full_path = os.path.join(save_dir, filename)
        print(f"准备保存图表到: {full_path}") # 更明确的日志
        try:
            plt.savefig(full_path)
            print(f"Matplotlib声称已保存图表到: {full_path}") # Matplotlib执行了保存
            
            # === 新增诊断代码 ===
            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path)
                print(f"确认：文件 {full_path} 确实存在。大小: {file_size} 字节。")
                if file_size == 0:
                    print(f"警告：文件 {full_path} 大小为0字节，可能为空白或损坏。")
            else:
                print(f"错误：文件 {full_path} 在保存后未能找到！请检查权限或路径问题。")
            # === 诊断代码结束 ===

        except Exception as e:
            print(f"保存图表 {full_path} 时发生错误: {e}")
    if show_plot:
        plt.show()
    plt.close() # Close plot to free memory

def plot_returns_comparison(strategy_results: dict, save_dir: str | None = None, show_plot: bool = True, suffix: str = ""):
    """
    Plots the cumulative returns of all strategies.

    Args:
        strategy_results (dict): Dictionary of strategy results,
                                 each containing 'portfolio_values'.
        save_dir (str, optional): Directory to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        suffix (str, optional): Suffix to append to the plot filename and title. Defaults to "".
    """
    plt.figure(figsize=(12, 6))
    for name, results in strategy_results.items():
        if 'portfolio_values' not in results or results['portfolio_values'] is None:
            print(f"警告 (plot_returns_comparison): 策略 '{name}' 缺少 'portfolio_values'。")
            continue
        
        portfolio_values = results['portfolio_values']
        if isinstance(portfolio_values, (list, np.ndarray)) and len(portfolio_values) < 1: # Need at least 1 value for initial
            print(f"警告 (plot_returns_comparison): 策略 '{name}' 的 'portfolio_values' 数据不足。")
            continue
        if not isinstance(portfolio_values, pd.Series):
             portfolio_values = pd.Series(portfolio_values)

        if portfolio_values.empty or portfolio_values.iloc[0] == 0:
            print(f"警告 (plot_returns_comparison): 策略 '{name}' 的初始投资组合价值为0或数据为空，无法计算回报率。")
            returns = pd.Series([0] * len(portfolio_values.index), index=portfolio_values.index) # Plot flat line at 0
        else:
            returns = portfolio_values / portfolio_values.iloc[0] - 1
        
        plt.plot(returns.index, returns, label=name) # Use index if Series

    title_suffix = _format_suffix_for_title(suffix)
    plt.title(f'Cumulative Return Comparison{title_suffix}')
    plt.xlabel('Trading Day / Time')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'returns_comparison{suffix}.png'
        plt.savefig(os.path.join(save_dir, filename))
        print(f"图表已保存到: {os.path.join(save_dir, filename)}")
    if show_plot:
        plt.show()
    plt.close()

def plot_benchmark_comparison(strategy_results: dict, benchmark_returns: pd.Series | None, 
                              save_dir: str | None = None, show_plot: bool = True, suffix: str = ""):
    """
    Plots strategy cumulative returns against benchmark cumulative returns.

    Args:
        strategy_results (dict): Dictionary of strategy results.
        benchmark_returns (pd.Series, optional): Series of benchmark daily returns.
                                                If None or empty, benchmark is not plotted.
        save_dir (str, optional): Directory to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        suffix (str, optional): Suffix to append to the plot filename and title. Defaults to "".
    """
    plt.figure(figsize=(12, 6))
    for name, results in strategy_results.items():
        if 'portfolio_values' not in results or results['portfolio_values'] is None:
            print(f"警告 (plot_benchmark_comparison): 策略 '{name}' 缺少 'portfolio_values'。")
            continue

        portfolio_values = results['portfolio_values']
        if isinstance(portfolio_values, (list, np.ndarray)) and len(portfolio_values) < 1:
            print(f"警告 (plot_benchmark_comparison): 策略 '{name}' 的 'portfolio_values' 数据不足。")
            continue
        if not isinstance(portfolio_values, pd.Series):
             portfolio_values = pd.Series(portfolio_values)

        if portfolio_values.empty or portfolio_values.iloc[0] == 0:
            print(f"警告 (plot_benchmark_comparison): 策略 '{name}' 的初始投资组合价值为0或数据为空，无法计算回报率。")
            returns = pd.Series([0] * len(portfolio_values.index), index=portfolio_values.index)
        else:
            returns = portfolio_values / portfolio_values.iloc[0] - 1
        
        plt.plot(returns.index, returns, label=name)

    if benchmark_returns is not None and not benchmark_returns.empty:
        # Assuming benchmark_returns are DAILY returns, calculate cumulative product
        # If benchmark_returns are already cumulative starting from 0, adjust accordingly.
        # For daily returns (e.g., output of calculate_benchmark_returns):
        cumulative_benchmark = (1 + benchmark_returns).cumprod() -1 # To start from 0 like other returns
        plt.plot(cumulative_benchmark.index, cumulative_benchmark, label='Benchmark (Buy & Hold Avg)', linestyle='--', color='k')
    else:
        print("提示 (plot_benchmark_comparison): 未提供基准回报或基准回报为空，将不绘制基准线。")

    title_suffix = _format_suffix_for_title(suffix)
    plt.title(f'Strategy Cumulative Return vs. Buy and Hold{title_suffix}')
    plt.xlabel('Trading Day / Time')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'benchmark_comparison{suffix}.png'
        full_path = os.path.join(save_dir, filename) # 获取完整路径
        
        print(f"准备保存图表到: {full_path}") # 更明确的日志
        try:
            plt.savefig(full_path)
            print(f"Matplotlib声称已保存图表到: {full_path}") # Matplotlib执行了保存
            
            # === 新增诊断代码 ===
            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path)
                print(f"确认：文件 {full_path} 确实存在。大小: {file_size} 字节。")
                if file_size == 0:
                    print(f"警告：文件 {full_path} 大小为0字节，可能为空白或损坏。")
            else:
                print(f"错误：文件 {full_path} 在保存后未能找到！请检查权限或路径问题。")
            # === 诊断代码结束 ===

        except Exception as e:
            print(f"保存图表 {full_path} 时发生错误: {e}")
    if show_plot:
        plt.show()
    plt.close()

def plot_all_baseline(strategy_results: dict, benchmark_returns: pd.Series | None, 
                      save_dir: str | None = None, suffix: str = ""):
    """
    Calls all individual plotting functions for baseline strategies.

    Args:
        strategy_results (dict): Dictionary of strategy results.
        benchmark_returns (pd.Series, optional): Series of benchmark daily returns.
        save_dir (str, optional): Directory to save the plots. Defaults to None.
        suffix (str, optional): Suffix to append to plot filenames. Defaults to "".
    """
    print(f"\n开始为后缀 '{suffix}' 生成基线策略图表...")
    if not strategy_results:
        print("警告 (plot_all_baseline): strategy_results 为空，无法生成图表。")
        return

    # Pass the suffix to each individual plotting function
    plot_portfolio_values(strategy_results, save_dir=save_dir, show_plot=False, suffix=suffix)
    plot_returns_comparison(strategy_results, save_dir=save_dir, show_plot=False, suffix=suffix)
    plot_benchmark_comparison(strategy_results, benchmark_returns, save_dir=save_dir, show_plot=False, suffix=suffix)
    print(f"后缀 '{suffix}' 的基线策略图表已生成完毕。")