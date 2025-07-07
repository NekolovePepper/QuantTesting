import pandas as pd
import numpy as np
import talib as ta
import os
import glob # For finding files

# --- MODIFICATION: Dynamic input and output directory setup ---
DATA_DIR = "data"
# Output file base name will be determined by input file type
# We no longer use a single INPUT_FILE or OUTPUT_FILE static variable here.

print("Starting feature engineering process...")

# Try to find actual data files first (e.g., stock_data_YYYY.csv)
input_file_pattern_actual = os.path.join(DATA_DIR, "stock_data_????.csv") # ???? for any 4-digit year
# Exclude files that already have "_features_" in their name to avoid reprocessing
all_potential_files = glob.glob(input_file_pattern_actual)
input_files = [f for f in all_potential_files if "_features_" not in os.path.basename(f) and "mock_" not in os.path.basename(f).lower()]
output_prefix = "stock_data_with_features"

if not input_files:
    # If no actual data files, try mock data files (e.g., mock_stock_data_YYYY.csv)
    print("No 'stock_data_YYYY.csv' files found (or they were already feature files). Looking for 'mock_stock_data_YYYY.csv' files.")
    input_file_pattern_mock = os.path.join(DATA_DIR, "mock_stock_data_????.csv")
    all_potential_mock_files = glob.glob(input_file_pattern_mock)
    input_files = [f for f in all_potential_mock_files if "_features_" not in os.path.basename(f)]
    output_prefix = "mock_stock_data_with_features"
    if input_files:
        print(f"Found mock data files: {input_files}")
    else:
        print(f"No 'mock_stock_data_YYYY.csv' files found either (or they were already feature files).")

if not input_files:
    print(f"Error: No suitable yearly data files found in '{DATA_DIR}' to process. Exiting.")
    exit()

print(f"Processing files with prefix '{output_prefix.replace('_with_features', '')}': {input_files}")

all_yearly_dfs = []
for f_path in input_files:
    try:
        print(f"Reading: {f_path}")
        temp_df = pd.read_csv(f_path)
        all_yearly_dfs.append(temp_df)
    except Exception as e:
        print(f"  Error reading {f_path}: {e}")

if not all_yearly_dfs:
    print("No data could be read from the files. Exiting.")
    exit()

# Combine all yearly data into a single DataFrame
df = pd.concat(all_yearly_dfs, ignore_index=True)

# Convert 'date' to datetime and sort values
# This is crucial for TA-Lib to calculate indicators correctly over time series
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by=['tic', 'date'], inplace=True)

print(f"\nCombined data shape: {df.shape}")
print("Combined columns:", df.columns.tolist())
print("Sample of combined data:")
print(df.head())
# --- END MODIFICATION: Dynamic input ---

all_stocks_with_features = []

for ticker, group in df.groupby('tic'):
    print(f"\nProcessing features for stock: {ticker}")
    
    # Data is already sorted by 'tic' and 'date' from the global sort
    # group = group.sort_values('date') # Not strictly necessary here if df was sorted
    
    # Get price data
    # Ensure data types are float for TA-Lib, handle potential non-numeric if any (though unlikely from prev script)
    close_prices = group['close'].values.astype(float)
    high_prices = group['high'].values.astype(float)
    low_prices = group['low'].values.astype(float)
    volume = group['volume'].values.astype(float)
    
    # Calculate Simple Moving Average (SMA)
    try:
        group['sma_5'] = ta.SMA(close_prices, timeperiod=5)
        group['sma_10'] = ta.SMA(close_prices, timeperiod=10)
        group['sma_20'] = ta.SMA(close_prices, timeperiod=20)
        group['sma_60'] = ta.SMA(close_prices, timeperiod=60)
        print(f"  Calculated SMA indicators for {ticker}")
    except Exception as e:
        print(f"  Error calculating SMA for {ticker}: {e}")
    
    # Calculate Relative Strength Index (RSI)
    try:
        group['rsi_14'] = ta.RSI(close_prices, timeperiod=14)
        print(f"  Calculated RSI indicator for {ticker}")
    except Exception as e:
        print(f"  Error calculating RSI for {ticker}: {e}")
    
    # Calculate Moving Average Convergence Divergence (MACD)
    try:
        macd, macd_signal, macd_hist = ta.MACD(
            close_prices, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        group['macd'] = macd
        group['macd_signal'] = macd_signal
        group['macd_hist'] = macd_hist
        print(f"  Calculated MACD indicators for {ticker}")
    except Exception as e:
        print(f"  Error calculating MACD for {ticker}: {e}")
    
    # Calculate Bollinger Bands
    try:
        upper, middle, lower = ta.BBANDS(
            close_prices, 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0 # SMA
        )
        group['bb_upper'] = upper
        group['bb_middle'] = middle
        group['bb_lower'] = lower
        print(f"  Calculated Bollinger Bands for {ticker}")
    except Exception as e:
        print(f"  Error calculating Bollinger Bands for {ticker}: {e}")
    
    # Calculate Average True Range (ATR) - volatility
    try:
        group['atr_14'] = ta.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        print(f"  Calculated ATR indicator for {ticker}")
    except Exception as e:
        print(f"  Error calculating ATR for {ticker}: {e}")
    
    # Calculate Rate of Change (ROC) - price momentum
    try:
        group['roc_10'] = ta.ROC(close_prices, timeperiod=10)
        print(f"  Calculated ROC indicator for {ticker}")
    except Exception as e:
        print(f"  Error calculating ROC for {ticker}: {e}")
    
    # Calculate Money Flow Index (MFI)
    try:
        group['mfi_14'] = ta.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
        print(f"  Calculated MFI indicator for {ticker}")
    except Exception as e:
        print(f"  Error calculating MFI for {ticker}: {e}")
    
    # Add other features: daily return
    # Ensure 'close' is float before pct_change
    group['daily_return'] = group['close'].astype(float).pct_change()
    
    all_stocks_with_features.append(group.copy()) # Use .copy() to avoid issues with SettingWithCopyWarning later

# Combine all stock data with features
if not all_stocks_with_features:
    print("\nNo features were calculated for any stock. Exiting.")
    exit()
    
df_with_features = pd.concat(all_stocks_with_features, ignore_index=False) # ignore_index=False to keep original index if needed, though group specific
df_with_features.reset_index(drop=True, inplace=True) # Reset index after concat

# Handle missing values (NaNs created by indicators at the start of series or by pct_change)
print("\nHandling missing values...")
missing_percent_before = df_with_features.isnull().mean() * 100
print("Missing value percentage per column (before handling):")
print(missing_percent_before[missing_percent_before > 0].sort_values(ascending=False))

# For technical indicators, forward fill is often appropriate.
# Then backfill for any remaining NaNs at the very beginning of the series.
df_with_features = df_with_features.fillna(method='ffill')
df_with_features = df_with_features.fillna(method='bfill')
# Any remaining NaNs (e.g., if a whole stock had insufficient data for an indicator) fill with 0.
df_with_features = df_with_features.fillna(0)

missing_percent_after = df_with_features.isnull().sum()
print("\nMissing values count per column (after handling, should be 0):")
print(missing_percent_after[missing_percent_after > 0])
if missing_percent_after.sum() == 0:
    print("All missing values handled.")

# --- MODIFICATION: Save data split by year ---
# Ensure 'date' column is datetime (should be already, but double check)
df_with_features['date'] = pd.to_datetime(df_with_features['date'])

unique_years = sorted(df_with_features['date'].dt.year.unique())
print(f"\nData processed for years: {unique_years}")

for year in unique_years:
    year_df = df_with_features[df_with_features['date'].dt.year == year].copy()
    
    # Construct output filename based on the determined prefix
    OUTPUT_FILE_YEAR = os.path.join(DATA_DIR, f"{output_prefix}_{year}.csv")
    
    year_df.to_csv(OUTPUT_FILE_YEAR, index=False)
    print(f"Saved features for year {year} to: {OUTPUT_FILE_YEAR} ({len(year_df)} rows)")
# --- END MODIFICATION: Save data split by year ---

print("\nFinal data shape (overall):", df_with_features.shape)
print("Final feature list:", df_with_features.columns.tolist())
print("\nFeature engineering complete!")