import pandas as pd
import os

print("Checking downloaded data...")

files = os.listdir("data/processed")

if not files:
    print("No data found. Please run get_data.py first.")
else:
    for file in files:
        if file.startswith("processed_"):
            df = pd.read_csv(f"data/processed/{file}")
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            stock = file.replace("processed_", "").replace(".csv", "")
            print(f"{stock}: {len(df)} records, from {df['timestamp'].min()} to {df['timestamp'].max()}")

print(f"\nTotal files processed: {len([f for f in files if f.startswith('processed_')])}")
