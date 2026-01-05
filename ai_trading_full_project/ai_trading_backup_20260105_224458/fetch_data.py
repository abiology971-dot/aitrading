import pandas as pd
import yfinance as yf

ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Flatten the multi-level column index if it exists
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Reset index to make Date a regular column
data = data.reset_index()

# Create target variable (1 if next day's close > today's close, 0 otherwise)
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

# Drop the last row since it has NaN target
data.dropna(inplace=True)

# Save to CSV
data.to_csv("stock_data.csv", index=False)

print("Data downloaded and saved")
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
