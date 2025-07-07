import pandas as pd
import numpy as np # 假设您的代码后续可能用到
import os
import glob
import re # 用于从文件名提取年份
from src.core.env import StockTradingEnv
# 假设 DATA_DIR 在函数外部定义，或者您可以将其作为参数传入
# DATA_DIR = "data" # 例如

def _extract_year_from_filename(filename: str) -> int | None:
    """(辅助函数) 从文件名中提取四位年份数字"""
    match = re.search(r'_(\d{4})\.csv$', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return None

def _find_available_feature_files_map(data_dir: str) -> dict[int, str]:
    """
    (辅助函数) 查找特征文件并返回一个年份到文件路径的映射。
    优先查找 "stock_data_with_features", 然后是 "mock_stock_data_with_features".
    """
    year_to_filepath_map = {}
    # 优先查找 stock_data_with_features
    pattern_actual = os.path.join(data_dir, "stock_data_with_features_????.csv")
    actual_files = glob.glob(pattern_actual)
    if actual_files:
        # print(f"发现 'stock_data_with_features' 系列文件。")
        for f_path in actual_files:
            year = _extract_year_from_filename(f_path)
            if year:
                year_to_filepath_map[year] = f_path
        return year_to_filepath_map

    # 如果未找到，尝试 mock_stock_data_with_features
    pattern_mock = os.path.join(data_dir, "mock_stock_data_with_features_????.csv")
    mock_files = glob.glob(pattern_mock)
    if mock_files:
        # print(f"未找到 'stock_data_with_features' 系列文件，发现 'mock_stock_data_with_features' 系列文件。")
        for f_path in mock_files:
            year = _extract_year_from_filename(f_path)
            if year:
                year_to_filepath_map[year] = f_path
        return year_to_filepath_map
    
    return year_to_filepath_map

def _load_data_for_specific_years(target_years: set[int], 
                                 available_files_map: dict[int, str], 
                                 dataset_label: str = "数据集") -> pd.DataFrame:
    """
    (辅助函数) 为指定的目标年份加载数据。
    """
    df_list = []
    loaded_years_for_this_set = []

    for year in sorted(list(target_years)): # 按年份顺序加载
        if year in available_files_map:
            file_path = available_files_map[year]
            # print(f"  为 {dataset_label} 加载年份 {year} 数据: {file_path}")
            try:
                df_list.append(pd.read_csv(file_path))
                loaded_years_for_this_set.append(year)
            except Exception as e:
                print(f"    读取 {file_path} 时出错: {e}")
        # else:
            # 这个警告在主函数中处理更合适，因为主函数知道哪些年份是“必须”的
            # print(f"  警告: {dataset_label} 的目标年份 {year} 的数据文件未找到。")

    if not df_list:
        # print(f"未能为 {dataset_label} (目标年份: {sorted(list(target_years))}) 加载任何数据。")
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    if 'date' in combined_df.columns:
        combined_df['date'] = pd.to_datetime(combined_df['date'])
    
    # print(f"  {dataset_label} (目标年份: {sorted(list(target_years))}) 加载完成, 包含实际年份: {sorted(loaded_years_for_this_set)}")
    return combined_df

def load_data(current_target_test_year: int):
    """
    加载数据，并根据给定的目标测试年份动态划分训练集和测试集。
    训练集为目标测试年份的前三年数据。
    测试集为目标测试年份当年的数据。

    返回:
        all_periods_data (pd.DataFrame): 所有已加载年份合并的数据。
        train_data (pd.DataFrame): 为当前目标测试年准备的训练数据。
        test_data (pd.DataFrame): 当前目标测试年的数据。
        ticker_list (list): 所有股票代码的列表。
    """
    global DATA_DIR 
    if 'DATA_DIR' not in globals() and 'DATA_DIR' not in locals():
        DATA_DIR = "data" 
        print(f"警告: DATA_DIR 未在 load_data 函数作用域内定义，临时设为 '{DATA_DIR}'")
        os.makedirs(DATA_DIR, exist_ok=True) # 确保目录存在

    print(f"\n--- 开始为目标测试年份 {current_target_test_year} 加载和划分数据 ---")

    available_files_map = _find_available_feature_files_map(DATA_DIR)

    if not available_files_map:
        print(f"错误: 在 '{DATA_DIR}' 目录中未找到任何年度特征文件。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    all_available_years_from_files = set(available_files_map.keys())
    print(f"数据目录中可用的数据年份: {sorted(list(all_available_years_from_files))}")

    # 1. 加载所有可用数据，构成总的 'all_periods_data' DataFrame
    all_periods_data = _load_data_for_specific_years(
        all_available_years_from_files, 
        available_files_map, 
        "总数据集(all_periods_data)"
    )
    
    if all_periods_data.empty:
        print("错误: 未能加载任何数据以形成总数据集。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    
    all_periods_data = all_periods_data.sort_values(['date', 'tic']) # 排序总数据
    
    ticker_list = all_periods_data['tic'].unique().tolist()
    print(f"股票列表 (来自总数据): {ticker_list}")
    # print(f"总数据日期范围: {all_periods_data['date'].min().date()} 到 {all_periods_data['date'].max().date()}")
    # print(f"总数据形状: {all_periods_data.shape}")

    # 2. 定义当前测试年份的训练和测试年份范围
    target_test_year_set = {current_target_test_year}
    target_train_years_set = set(range(current_target_test_year - 3, current_target_test_year))
    
    print(f"为目标测试年份 {current_target_test_year}，定义的训练年份范围: {sorted(list(target_train_years_set))}")
    print(f"为目标测试年份 {current_target_test_year}，定义的目标测试年份: {current_target_test_year}")

    # 3. 从 all_periods_data 中筛选训练数据
    #   检查这些目标训练年份的数据是否真的在已加载的总数据中 (即文件是否存在)
    actual_train_years_to_use = target_train_years_set.intersection(all_available_years_from_files)
    
    if not actual_train_years_to_use:
        print(f"警告: 目标训练年份 {sorted(list(target_train_years_set))} 的数据均未在可用文件中找到。训练集将为空。")
        train_data = pd.DataFrame()
    else:
        if len(actual_train_years_to_use) < len(target_train_years_set):
            missing_train_y = sorted(list(target_train_years_set - actual_train_years_to_use))
            print(f"警告: 目标训练年份中的 {missing_train_y} 数据文件未找到。训练集仅包含年份 {sorted(list(actual_train_years_to_use))}.")
        
        # 直接从已加载的 all_periods_data 中筛选
        train_data = all_periods_data[all_periods_data['date'].dt.year.isin(actual_train_years_to_use)].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        if not train_data.empty:
            # train_data 已是排序的子集，无需再次排序
            print(f"训练集 (for test year {current_target_test_year}) 已提取。实际包含年份: {sorted(list(train_data['date'].dt.year.unique()))}。日期范围: {train_data['date'].min().date()} 到 {train_data['date'].max().date()}。形状: {train_data.shape}")
        else:
             print(f"训练集 (for test year {current_target_test_year}) 为空，即使部分目标训练年份 {sorted(list(actual_train_years_to_use))} 有文件 (可能数据为空或筛选问题)。")

    # 4. 从 all_periods_data 中筛选测试数据
    actual_test_years_to_use = target_test_year_set.intersection(all_available_years_from_files)

    if not actual_test_years_to_use: # 即 current_target_test_year 的数据文件不存在
        print(f"警告: 目标测试年份 {current_target_test_year} 的数据文件未找到。测试集将为空。")
        test_data = pd.DataFrame()
    else:
        # 直接从已加载的 all_periods_data 中筛选
        test_data = all_periods_data[all_periods_data['date'].dt.year.isin(actual_test_years_to_use)].copy() # 使用 .copy()
        if not test_data.empty:
            # test_data 已是排序的子集
            print(f"测试集 (for test year {current_target_test_year}) 已提取。实际包含年份: {sorted(list(test_data['date'].dt.year.unique()))}。日期范围: {test_data['date'].min().date()} 到 {test_data['date'].max().date()}。形状: {test_data.shape}")
        else:
            print(f"测试集 (for test year {current_target_test_year}) 为空，即使目标年份 {current_target_test_year} 有文件 (可能数据为空或筛选问题)。")
            
    return all_periods_data, train_data, test_data, ticker_list

# 假设 StockTradingEnv 和 EnhancedRewardStockTradingEnv 类定义在上方或已导入
# from src.core.enhanced_env import EnhancedRewardStockTradingEnv # 如果您有这个类

def create_environments(train_data: pd.DataFrame, 
                        test_data: pd.DataFrame, 
                        global_ticker_list: list, # 参数名改为 global_ticker_list 以强调其来源
                        tech_indicators: list, 
                        use_enhanced_reward: bool = False):

    """创建训练和测试环境"""
    EnvClass = EnhancedRewardStockTradingEnv if use_enhanced_reward else StockTradingEnv
    # 调试断点，检查传入的参数
    train_env = None
    if not train_data.empty:
        # 确保 train_data 中的日期是唯一的，并且至少有几天数据可以运行
        if train_data['date'].nunique() > 1: # 至少需要2天才能进行一步step（当前天和下一天）
            train_env = EnvClass(
                df=train_data,
                stock_dim=len(global_ticker_list)-1,      # 使用全局列表的长度
                global_ticker_list=global_ticker_list,  # 传入全局股票列表
                hmax=10,                                # 您原来的参数
                initial_amount=1000000,
                transaction_cost_pct=0.001,
                reward_scaling=1e-4,
                tech_indicator_list=tech_indicators
            )
        else:
            print("警告 (create_environments): 训练数据天数不足 (<2)，无法创建有效的训练环境。")
    else:
        print("警告 (create_environments): 训练数据为空，无法创建训练环境。")

    test_env = None
    if not test_data.empty:
        if test_data['date'].nunique() > 1: # 至少需要2天
            test_env = EnvClass(
                df=test_data,
                stock_dim=len(global_ticker_list),       # 使用全局列表的长度
                global_ticker_list=global_ticker_list,   # 传入全局股票列表
                hmax=10,
                initial_amount=1000000,
                transaction_cost_pct=0.001,
                reward_scaling=1e-4,
                tech_indicator_list=tech_indicators
            )
        else:
            print("警告 (create_environments): 测试数据天数不足 (<2)，无法创建有效的测试环境。")
    else:
        print("警告 (create_environments): 测试数据为空，无法创建测试环境。")
        
    return train_env, test_env

# --- 示例用法 (需要确保 DATA_DIR 和相关文件存在) ---
if __name__ == '__main__':
    # 为了测试，我们假设 DATA_DIR 和一些模拟文件存在
    if 'DATA_DIR' not in globals(): # 确保 DATA_DIR 被定义
        DATA_DIR = "data_test_rolling_window" 
    
    os.makedirs(DATA_DIR, exist_ok=True)

    sample_tickers = ['STOCK_A', 'STOCK_B']
    base_columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', 'feature1', 'feature2']
    
    def create_dummy_year_file_for_test(year, tickers, data_dir):
        file_path = os.path.join(data_dir, f"stock_data_with_features_{year}.csv") # 使用此命名
        # 为一年中的每个月创建几天数据，以模拟更真实的时间跨度
        all_data_for_year = []
        for month in range(1, 13):
            try:
                start_date_str = f'{year}-{month:02d}-01'
                # 创建3个工作日的数据
                dates = pd.date_range(start=start_date_str, periods=3, freq='B') 
            except ValueError: # 处理像2月30日这样的无效日期（尽管这里不会发生）
                continue

            for ticker in tickers:
                df_ticker_month = pd.DataFrame(index=dates)
                df_ticker_month['date'] = df_ticker_month.index
                df_ticker_month['tic'] = ticker
                for col in ['open', 'high', 'low', 'close']:
                    df_ticker_month[col] = np.random.rand(len(dates)) * 100 + (50 + year % 10) # 价格随年份略微变化
                df_ticker_month['volume'] = np.random.randint(100000, 1000000, size=len(dates))
                df_ticker_month['feature1'] = np.random.randn(len(dates))
                df_ticker_month['feature2'] = np.random.rand(len(dates)) * 10
                all_data_for_year.append(df_ticker_month[base_columns])
        
        if all_data_for_year:
            final_df_year = pd.concat(all_data_for_year)
            final_df_year.to_csv(file_path, index=False)
            # print(f"创建了模拟文件: {file_path}，包含 {len(final_df_year)} 行")
        # else:
            # print(f"未能为年份 {year} 创建模拟数据。")

    print("正在创建模拟年度特征数据文件 (2016-2023)...")
    for year_to_create in range(2015, 2024): # 创建 2015 到 2023 年的数据, 确保有足够历史数据
        create_dummy_year_file_for_test(year_to_create, sample_tickers, DATA_DIR)
    print("模拟文件创建完成。")

    # 测试滚动窗口加载
    test_years_for_model = [2019, 2020, 2021, 2022, 2023] # 您希望测试的年份
    
    for target_year in test_years_for_model:
        all_data, train_set, test_set, tickers = load_data(target_year)
        
        print(f"--- 结果 for target_test_year: {target_year} ---")
        if not train_set.empty:
            print(f"  训练集年份: {sorted(list(train_set['date'].dt.year.unique()))}")
            print(f"  训练集大小: {train_set.shape}")
        else:
            print(f"  训练集为空 for target_test_year: {target_year}")
            
        if not test_set.empty:
            print(f"  测试集年份: {sorted(list(test_set['date'].dt.year.unique()))}")
            print(f"  测试集大小: {test_set.shape}")
        else:
            print(f"  测试集为空 for target_test_year: {target_year}")
        print("--- End of results ---")

    # 清理模拟文件和目录 (可选)
    # import shutil
    # if os.path.exists(DATA_DIR) and DATA_DIR == "data_test_rolling_window":
    #     shutil.rmtree(DATA_DIR)
    #     print(f"\n已删除模拟数据目录: {DATA_DIR}")s