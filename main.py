import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from src.rl.PPO import PPOAgent # 假设PPOAgent在src/rl/PPO.py中定义

# 假设这些导入路径都是正确的，并且模块功能符合预期
# sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 通常在项目根目录运行脚本时不需要
from src.core.env import StockTradingEnv
from src.core.enhanced_env import EnhancedRewardStockTradingEnv
from src.core.strategies import (
    equal_weight_strategy,
    price_inverse_strategy,
    momentum_strategy,
    mean_reversion_strategy,
    volume_weighted_strategy
)
from src.evaluation.evaluation import (
    evaluate_strategy,
    calculate_benchmark_returns,
    create_metrics_table,
    # update_strategy_metrics # 暂时注释，如果它的逻辑复杂或有问题
)
from src.plot.baseline_plot import plot_all_baseline
# from src.rl.train_rl_agent import main as rl_train_main_entry # 重命名以避免与此处的main冲突
# from src.hyperparameter.hyperparameter_tuning import HyperparameterTuner

# === 1. 路径设置 (保持简洁，确保正确) ===
# 获取 main.py 脚本所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "strategy_eval") # 给策略评估一个子目录
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === 2. 导入您已经写好的 load_data 函数 ===
# 假设它位于 src/utils/load_data.py 并且签名是 load_data(current_target_test_year, data_dir)
try:
    from src.utils.load_data import load_data, create_environments
    print("成功从 src.utils.load_data 导入 load_data 函数。")
except ImportError:
    print("错误: 无法从 src.utils.load_data 导入 load_data 函数。请确保该文件和函数存在且路径正确。")
    # 定义一个占位符函数，以便脚本的其余部分至少可以进行语法检查
    def load_data(current_target_test_year: int, data_dir: str):
        print(f"警告: 正在使用占位符 load_data({current_target_test_year}, '{data_dir}')。请实现真实的 load_data。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    # sys.exit(1) # 或者直接退出


# === 4. 策略评估函数 (针对单个窗口) ===
def run_strategy_evaluation_for_window(target_test_year: int, 
                                       test_env: StockTradingEnv, 
                                       test_df_for_benchmark: pd.DataFrame, 
                                       ticker_list: list, 
                                       tech_indicators: list):
    """评估所有基线策略 (针对特定窗口的 test_env 和 test_df)。"""
    if test_env is None or test_df_for_benchmark.empty:
        print(f"错误 (run_strategy_evaluation_for_window): 测试环境或测试数据为空 for year {target_test_year}，无法评估。")
        return {}, pd.Series(dtype=float) # 返回空结果

    strategies = {
        "等权重策略": equal_weight_strategy,
        "价格反向策略": price_inverse_strategy,
        "动量策略": momentum_strategy,
        "均值回归策略": mean_reversion_strategy,
        "成交量加权策略": volume_weighted_strategy
    }
    strategy_results_this_window = {}
    
    for name, strategy_func in strategies.items():
        strategy_name_suffixed = f"{name}_TestYear_{target_test_year}"
        print(f"\n  评估策略: {strategy_name_suffixed}...")
        
        # 确保 evaluate_strategy 被正确调用
        # evaluate_strategy 应该在 env 上运行，env 已经包含了正确的 df (即 test_df_for_benchmark)
        results = evaluate_strategy(
            strategy_name=strategy_name_suffixed,
            strategy_function=strategy_func,
            env=test_env, # test_env 是为当前 target_test_year 的 test_data 创建的
            n_stocks=len(ticker_list),
            tech_indicators=tech_indicators 
        )
        strategy_results_this_window[strategy_name_suffixed] = results
    
    benchmark_returns_this_window = calculate_benchmark_returns(test_df_for_benchmark, ticker_list)
    
    if strategy_results_this_window:
        metrics_table_this_window = create_metrics_table(strategy_results_this_window)
        print(f"\n  策略性能指标 (测试年份: {target_test_year}):")
        print(metrics_table_this_window)
        metrics_table_this_window.to_csv(os.path.join(RESULTS_DIR, f'metrics_strategy_testyear_{target_test_year}.csv'))
        
        # 为当前窗口绘图
        plot_all_baseline(strategy_results_this_window, 
                          benchmark_returns_this_window, 
                          RESULTS_DIR, 
                          suffix=f"_testyear_{target_test_year}")
    
    return strategy_results_this_window, benchmark_returns_this_window


# === 5. RL 相关函数 (简化版，标记为需要用户根据其 RL 框架适配) ===
def run_rl_train_for_window(target_test_year: int, train_env: StockTradingEnv, ticker_list: list):
    """(示例) 为给定窗口训练RL模型。用户需根据自己的RL框架修改此函数。"""
    if train_env is None:
        print(f"错误 (run_rl_train_for_window): 训练环境为空 for year {target_test_year}，无法训练。")
        return
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
    # 这里应该是您调用RL训练逻辑的地方，例如：
    from src.rl.train_rl_agent import  train_agent

    agent_name = f"ppo_agent_testyear_{target_test_year}"
    model_save_path = os.path.join(MODEL_DIR, f"agent_for_testyear_{target_test_year}.zip")
    training_history = train_agent(
        agent=agent,
        env=train_env,
        n_episodes=50,
        max_steps=200,
        save_freq=10
    )
    
    # 保存新模型（用不同文件名）
    model_save_path = os.path.join(MODEL_DIR, f"ppo_agent_{agent_name}.pth")
    agent.save(MODEL_DIR, f"ppo_agent_{agent_name}")
    print(f"新模型已保存至 {model_save_path}")
    


def run_rl_evaluate_for_window(target_test_year: int, test_env: StockTradingEnv, test_df: pd.DataFrame, ticker_list: list):
    """ 为给定窗口评估RL模型。用户需根据自己的RL框架修改此函数。"""
    if test_env is None or test_df.empty:
        print(f"错误 (run_rl_evaluate_for_window): 测试环境或数据为空 for year {target_test_year}，无法评估RL模型。")
        return {}
    print(f"  象征性RL评估 for target_test_year {target_test_year} using test_env...")
    # 这里应该是您调用RL评估逻辑的地方，例如：
    from src.rl.PPO import PPOAgent # 或您的其他代理
    from src.rl.train_rl_agent import evaluate_agent # 假设的评估函数
    model_load_path = os.path.join(MODEL_DIR, f"agent_for_testyear_{target_test_year}.zip")

    agent = PPOAgent.load(directory=model_load_path) # 示例加载
        # 评估代理
    rl_results = evaluate_agent(agent, test_env, ticker_list, test_df)
    return {f"RL_Agent_TestYear_{target_test_year}": rl_results}

# === 6. 主函数 ===
def main():
    parser = argparse.ArgumentParser(description="投资组合选择框架命令行工具 (滚动窗口版)")
    parser.add_argument("mode", choices=["strategy_test_year", "full_rolling_eval", "train_rl_year", "eval_rl_year"], 
                        help="运行模式: "
                             "'strategy_test_year' (对单个目标测试年评估基线策略), "
                             "'full_rolling_eval' (对指定年份范围进行完整的滚动评估，包括基线和RL), "
                             "'train_rl_year' (对单个目标测试年训练RL模型), "
                             "'eval_rl_year' (对单个目标测试年评估RL模型)")
    parser.add_argument("--target_year", type=int, default=2019, 
                        help="目标测试年份 (用于 strategy_test_year, train_rl_year, eval_rl_year 模式)")
    parser.add_argument("--start_year", type=int, default=2019, 
                        help="滚动评估的起始测试年份 (用于 full_rolling_eval 模式)")
    parser.add_argument("--end_year", type=int, default=2023, 
                        help="滚动评估的结束测试年份 (用于 full_rolling_eval 模式)")
    args = parser.parse_args()

    tech_indicators = ['sma_5', 'sma_10', 'sma_20', 'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr_14']

    if args.mode == "strategy_test_year":
        print(f"[模式: strategy_test_year] 评估基线策略 for 目标测试年份: {args.target_year}...")
        
        # 调用 load_data (它现在是您外部定义的滚动窗口版本)
        # 传递全局 DATA_DIR
        _, _, current_test_data, current_ticker_list = load_data(args.target_year, DATA_DIR)

        if current_test_data.empty or not current_ticker_list:
            print(f"  无法为年份 {args.target_year} 加载有效测试数据或股票列表，评估中止。")
        else:
            # 创建测试环境 (注意：这里不创建训练环境，因为仅评估)
            _, current_test_env = create_environments(pd.DataFrame(), current_test_data, current_ticker_list, tech_indicators)
            
            if current_test_env:
                run_strategy_evaluation_for_window(
                    args.target_year, current_test_env, current_test_data, current_ticker_list, tech_indicators
                )
                print(f"  基线策略评估 for {args.target_year} 完成。")
            else:
                print(f"  无法为年份 {args.target_year} 创建测试环境，评估中止。")

    elif args.mode == "train_rl_year":
        print(f"[模式: train_rl_year] 训练RL模型 for 目标测试年份: {args.target_year} (使用其前三年数据)...")
        _, current_train_data, current_test_data, current_ticker_list = load_data(args.target_year)
        if current_train_data.empty or not current_ticker_list:
            print(f"  无法为年份 {args.target_year} 加载有效训练数据或股票列表，训练中止。")
        else:
            current_train_env, _ = create_environments(current_train_data, current_test_data, current_ticker_list, tech_indicators)
            if current_train_env:
                run_rl_train_for_window(args.target_year, current_train_env, current_ticker_list)
                print(f"  RL模型训练 for {args.target_year} 完成。")
                run_rl_evaluate_for_window(args.target_year, current_train_env, current_train_data, current_ticker_list)
            else:
                print(f"  无法为年份 {args.target_year} 创建训练环境，训练中止。")


    elif args.mode == "eval_rl_year":
        print(f"[模式: eval_rl_year] 评估RL模型 for 目标测试年份: {args.target_year}...")
        _, _, current_test_data, current_ticker_list = load_data(args.target_year, DATA_DIR)
        if current_test_data.empty or not current_ticker_list:
            print(f"  无法为年份 {args.target_year} 加载有效测试数据或股票列表，RL评估中止。")
        else:
            _, current_test_env = create_environments(pd.DataFrame(), current_test_data, current_ticker_list, tech_indicators)
            if current_test_env:
                rl_results = run_rl_evaluate_for_window(args.target_year, current_test_env, current_test_data, current_ticker_list)
                if rl_results: # 如果评估函数返回了结果
                    print(f"  RL模型评估 for {args.target_year} 完成。策略: {list(rl_results.keys())}")
                    # 这里可以添加对 rl_results 的进一步处理，例如保存指标、绘图等
                    # metrics_table_rl = create_metrics_table(rl_results)
                    # print(metrics_table_rl)
                    # plot_all_baseline(rl_results, None, RESULTS_DIR, suffix=f"_rl_testyear_{args.target_year}")
            else:
                print(f"  无法为年份 {args.target_year} 创建测试环境，RL评估中止。")


    elif args.mode == "full_rolling_eval":
        print(f"[模式: full_rolling_eval] 运行完整滚动窗口评估 from {args.start_year} to {args.end_year}...")
        all_windows_combined_results = {} # 用于收集所有窗口的所有策略结果

        for year_to_test_loop in range(args.start_year, args.end_year + 1):
            print(f"\n{'='*20} 滚动窗口: 目标测试年份 {year_to_test_loop} {'='*20}")
            
            all_data, train_df, test_df, ticker_list_loop = load_data(year_to_test_loop)

            if test_df.empty or not ticker_list_loop:
                print(f"  测试年份 {year_to_test_loop} 的数据为空或股票列表为空，跳过此窗口。")
                continue
            
            train_env_loop, test_env_loop = create_environments(train_df, test_df, ticker_list_loop, tech_indicators)

            if test_env_loop is None: # 主要关注测试环境是否存在以进行评估
                print(f"  无法为测试年份 {year_to_test_loop} 创建测试环境，跳过此窗口的评估。")
                continue

            # (可选步骤) 在此窗口重新训练RL模型
            # print(f"\n  --- (步骤1 - 可选) 为 {year_to_test_loop} 训练RL模型 ---")
            if train_env_loop:
                run_rl_train_for_window(year_to_test_loop, train_env_loop, ticker_list_loop)
            else:
                print(f"  由于训练环境为空，跳过为 {year_to_test_loop} 训练RL模型。")


            print(f"\n  --- (步骤2) 为 {year_to_test_loop} 评估基线策略 ---")
            strategy_res, benchmark_res = run_strategy_evaluation_for_window(
                year_to_test_loop, test_env_loop, test_df, ticker_list_loop, tech_indicators
            )
            if strategy_res:
                all_windows_combined_results.update(strategy_res)

            print(f"\n  --- (步骤3) 为 {year_to_test_loop} 评估RL模型 ---")
            rl_res = run_rl_evaluate_for_window(year_to_test_loop, test_env_loop, test_df, ticker_list_loop)
            if rl_res:
                all_windows_combined_results.update(rl_res)
        
        if all_windows_combined_results:
            print("\n\n======== 滚动评估总汇报告 ========")
            final_summary_table = create_metrics_table(all_windows_combined_results)
            print("\n所有滚动窗口的综合性能指标:")
            print(final_summary_table)
            summary_filename = os.path.join(RESULTS_DIR, 'summary_all_rolling_windows_metrics.csv')
            final_summary_table.to_csv(summary_filename)
            print(f"综合性能指标已保存到: {summary_filename}")
            # 汇总绘图比较复杂，plot_all_baseline 可能需要修改或针对汇总结果特殊处理
        else:
            print("完整滚动评估未产生任何可汇总的结果。")
            
    # 'tune' 模式的适配较为复杂，此处保持原样并提示
    elif args.mode == "tune":
        print("警告: 超参数调优 (tune) 模式尚未针对滚动窗口完全适配。")
        from src.hyperparameter.hyperparameter_tuning import HyperparameterTuner # 确保导入
        run_hyperparameter_tuning() # 调用原函数
    else:
        print(f"错误: 未知模式 '{args.mode}'。")
        parser.print_help()

if __name__ == "__main__":
    # 模拟数据创建（用于测试，实际运行时应有真实数据）
    print(f"检查/创建模拟数据文件在: {DATA_DIR} (仅用于测试 main 函数)...")
    os.makedirs(DATA_DIR, exist_ok=True) # 确保 DATA_DIR 存在
    def _create_dummy_year_file_for_main_test(year, tickers, data_dir_local):
        file_path = os.path.join(data_dir_local, f"stock_data_with_features_{year}.csv")
        if os.path.exists(file_path) and os.path.getsize(file_path) > 100: return
        
        all_data_for_year_dummy = []
        base_cols_dummy = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', 
                           'sma_5', 'sma_10', 'sma_20', 'rsi_14', 'macd', 'macd_signal', 
                           'bb_upper', 'bb_lower', 'atr_14']
        for month_dummy in range(1, 13):
            try: dates_dummy = pd.date_range(start=f'{year}-{month_dummy:02d}-01', periods=3, freq='B') 
            except ValueError: continue
            for ticker_dummy in tickers:
                df_month_dummy = pd.DataFrame(index=dates_dummy)
                df_month_dummy['date'] = df_month_dummy.index.strftime('%Y-%m-%d')
                df_month_dummy['tic'] = ticker_dummy
                for col_dummy in ['open', 'high', 'low', 'close']: df_month_dummy[col_dummy] = np.random.rand(len(dates_dummy)) * 100 + 50
                df_month_dummy['volume'] = np.random.randint(100000, 1000000, size=len(dates_dummy))
                for ti_col_dummy in base_cols_dummy:
                    if ti_col_dummy not in df_month_dummy.columns: df_month_dummy[ti_col_dummy] = np.random.rand(len(dates_dummy))
                all_data_for_year_dummy.append(df_month_dummy[base_cols_dummy])
        if all_data_for_year_dummy: pd.concat(all_data_for_year_dummy).to_csv(file_path, index=False)
    
  
    print("模拟数据文件检查/创建完成。")
    
    main()