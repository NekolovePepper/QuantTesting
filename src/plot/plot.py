import os
import matplotlib.pyplot as plt
import numpy as np

def plot_portfolio_values(strategy_results, save_dir=None, show_plot=True):
    plt.figure(figsize=(12, 6))
    for name, results in strategy_results.items():
        portfolio_values = results['portfolio_values']
        plt.plot(portfolio_values, label=name)
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'portfolio_value_comparison.png'))
    if show_plot:
        plt.show()

def plot_returns_comparison(strategy_results, save_dir=None, show_plot=True):
    plt.figure(figsize=(12, 6))
    for name, results in strategy_results.items():
        portfolio_values = results['portfolio_values']
        returns = np.array(portfolio_values) / portfolio_values[0] - 1
        plt.plot(returns, label=name)
    plt.title('Cumulative Return Comparison')
    plt.xlabel('Trading Day')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'returns_comparison.png'))
    if show_plot:
        plt.show()

def plot_benchmark_comparison(strategy_results, benchmark_returns, save_dir=None, show_plot=True):
    plt.figure(figsize=(12, 6))
    for name, results in strategy_results.items():
        portfolio_values = results['portfolio_values']
        returns = np.array(portfolio_values) / portfolio_values[0] - 1
        plt.plot(returns, label=name)
    plt.plot(benchmark_returns, label='Buy and Hold')
    plt.title('Strategy Return vs. Buy and Hold')
    plt.xlabel('Trading Day')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'benchmark_comparison.png'))
    if show_plot:
        plt.show()

def plot_all(strategy_results, benchmark_returns, save_dir=None):
    plot_portfolio_values(strategy_results, save_dir=save_dir, show_plot=False)
    plot_returns_comparison(strategy_results, save_dir=save_dir, show_plot=False)
    plot_benchmark_comparison(strategy_results, benchmark_returns, save_dir=save_dir, show_plot=False)
