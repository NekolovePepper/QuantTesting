import pandas as pd
import pandas_datareader as pdr
import os
import time
from datetime import datetime
import numpy as np # Moved here, was used in mock data

# 创建数据目录
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 选择一小部分股票用于测试 (THIS IS YOUR ORIGINAL LIST)
TICKER_LIST = ["AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WMT"]
print(f"开始使用 Stooq 下载数据，股票列表: {TICKER_LIST}")

start_date = datetime(2016, 1, 1)
end_date = datetime(2023, 12, 31)

# 逐个下载股票数据
all_data = []
for ticker in TICKER_LIST:
    print(f"\n正在处理 {ticker}...")
    try:
        # 使用 Stooq 数据源
        ticker_data = pdr.stooq.StooqDailyReader(ticker, start=start_date, end=end_date).read()
        
        if not ticker_data.empty:
            ticker_data['tic'] = ticker  # 添加股票代码列
            ticker_data.reset_index(inplace=True)  # 将日期从索引转为列
            all_data.append(ticker_data)
            print(f"{ticker} 数据下载成功，获取了 {len(ticker_data)} 行")
        else:
            print(f"{ticker} 返回空数据")
            
    except Exception as e:
        print(f"{ticker} 下载过程中出错: {e}")
    
    # 为避免限制，每次请求之间暂停几秒
    if ticker != TICKER_LIST[-1]:
        wait_time = 2
        print(f"等待 {wait_time} 秒后继续...")
        time.sleep(wait_time)

# 合并和处理数据
combined_df = pd.DataFrame() # Initialize to handle case where all_data might be empty

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 重命名列以符合FinRL的要求
    rename_dict = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }
    # 仅重命名存在的列
    rename_dict = {k: v for k, v in rename_dict.items() if k in combined_df.columns}
    combined_df.rename(columns=rename_dict, inplace=True, errors='ignore')
    
    # 确保 'date' 列是 datetime 类型
    if 'date' in combined_df.columns:
        combined_df['date'] = pd.to_datetime(combined_df['date'])
    else:
        print("错误: 'date' 列在下载的数据中不存在或重命名失败。")
        # This will likely lead to the mock data section if 'date' is crucial and missing
        combined_df = pd.DataFrame() # Ensure it's empty to trigger mock data if 'date' missing

    if not combined_df.empty and 'date' in combined_df.columns:
        # 添加adjcp列
        if 'adjcp' not in combined_df.columns and 'close' in combined_df.columns:
            combined_df['adjcp'] = combined_df['close']  # Stooq数据通常没有调整收盘价
        
        # 确保所有必要列存在
        essential_columns = ['date', 'open', 'high', 'low', 'close', 'adjcp', 'volume', 'tic']
        available_columns = [col for col in essential_columns if col in combined_df.columns]
        
        # 只保留必要的列
        combined_df = combined_df[available_columns]
        
        print(f"\n数据下载并合并完成。总行数: {len(combined_df)}")
        print("合并数据示例:")
        print(combined_df.head())

        # --- MODIFICATION START: Save data for each year separately ---
        unique_years = sorted(combined_df['date'].dt.year.unique())
        print(f"\n发现年份: {unique_years}")
        for year in unique_years:
            year_df = combined_df[combined_df['date'].dt.year == year].copy() # Use .copy() to avoid SettingWithCopyWarning
            # Sort data by date and ticker within each year's file for consistency
            year_df.sort_values(by=['date', 'tic'], inplace=True)
            OUTPUT_FILE_YEAR = os.path.join(DATA_DIR, f"stock_data_{year}.csv")
            year_df.to_csv(OUTPUT_FILE_YEAR, index=False)
            print(f"年份 {year} 的数据已保存到 {OUTPUT_FILE_YEAR} ({len(year_df)} 行)")
        # --- MODIFICATION END ---
    else:
        print("\n合并数据处理失败或 'date' 列缺失，将尝试创建模拟数据。")
        all_data = [] # Ensure this is consistent for the next check

if not all_data or combined_df.empty or 'date' not in combined_df.columns: # Check if we need to generate mock data
    print("\n没有成功下载或处理任何有效数据（或 'date' 列缺失），创建模拟数据")
    
    # 创建模拟数据作为备选方案
    # import numpy as np # Already imported at the top
    from datetime import datetime, timedelta # datetime already imported, timedelta needed
    
    # 创建日期范围
    dates_mock = pd.date_range(start='2018-01-01', end='2023-12-31', freq='B')  # 工作日
    
    mock_data_list = [] # Renamed to avoid conflict
    for ticker_mock in TICKER_LIST: # Use the original TICKER_LIST
        # 初始价格范围
        if ticker_mock in ["MSFT", "AAPL"]: # Original logic for price generation
            price = np.random.uniform(100, 200)
        else:
            price = np.random.uniform(50, 150)
        
        for date_val_mock in dates_mock: # Renamed to avoid conflict
            # 随机每日变动
            change = np.random.normal(0.0005, 0.015)  # 均值略大于0表示长期上涨
            price *= (1 + change)
            price = max(0.01, price) # Ensure price doesn't go to zero or negative
            
            # 生成日内其他价格
            open_price = price * (1 + np.random.uniform(-0.01, 0.01))
            high_price = max(price, open_price) * (1 + np.random.uniform(0, 0.02))
            low_price = min(price, open_price) * (1 - np.random.uniform(0, 0.02))
            volume = int(np.random.uniform(100000, 10000000))
            
            mock_data_list.append({
                'date': date_val_mock,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': price,
                'adjcp': price,
                'volume': volume,
                'tic': ticker_mock
            })
    
    mock_df = pd.DataFrame(mock_data_list)
    
    print(f"\n模拟数据已生成。总行数: {len(mock_df)}")
    print("模拟数据示例:")
    print(mock_df.head())
    
    # --- MODIFICATION START: Save mock data for each year separately ---
    if 'date' in mock_df.columns: # Should always be true here
        mock_df['date'] = pd.to_datetime(mock_df['date']) # Ensure datetime format
        unique_years_mock = sorted(mock_df['date'].dt.year.unique())
        print(f"\n模拟数据中的年份: {unique_years_mock}")
        for year_mock in unique_years_mock:
            year_mock_df = mock_df[mock_df['date'].dt.year == year_mock].copy() # Use .copy()
            # Sort data by date and ticker within each year's file for consistency
            year_mock_df.sort_values(by=['date', 'tic'], inplace=True)
            MOCK_FILE_YEAR = os.path.join(DATA_DIR, f"mock_stock_data_{year_mock}.csv")
            year_mock_df.to_csv(MOCK_FILE_YEAR, index=False)
            print(f"年份 {year_mock} 的模拟数据已保存到 {MOCK_FILE_YEAR} ({len(year_mock_df)} 行)")
    # --- MODIFICATION END ---

print("\n脚本执行完成")