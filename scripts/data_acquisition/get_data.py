import yfinance as yf
import pandas as pd
import numpy as np
import os

print("Starting data download...")

# Create folders
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Stock symbols to download
stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']

print(f"Downloading data for: {stocks}")

for stock in stocks:
    print(f"Downloading {stock}...")
    
    # Get the stock data (1 year of daily prices)
    data = yf.download(stock, start="2023-01-01", end="2024-01-01")
    
    # Save raw data
    data.to_csv(f"data/raw/{stock}_daily.csv")
    
    # Process the data for our analysis
    df = data.reset_index()
    df['return'] = (df['Close'] - df['Open']) / df['Open']
    df['side'] = np.where(df['return'] > 0, 1, -1)  # 1 for up, -1 for down
    df['timestamp'] = pd.to_datetime(df['Date'])
    
    # Save processed data
    df.to_csv(f"data/processed/processed_{stock}.csv", index=False)
    
    print(f"✓ {stock}: {len(df)} days of data")

print("All data downloaded successfully!")
print("Raw data saved in: data/raw/")
print("Processed data saved in: data/processed/")
