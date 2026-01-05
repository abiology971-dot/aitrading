#!/usr/bin/env python3
"""
Quick test script for all trading models
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("=" * 70)
print("AI Trading Project - Quick Model Testing")
print("=" * 70)

# Load data
print("\n1. Loading stock data...")
try:
    data = pd.read_csv("stock_data.csv")
    print(f"   ✓ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"   ✓ Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(
        f"   ✓ Target distribution: UP={sum(data['Target'] == 1)}, DOWN={sum(data['Target'] == 0)}"
    )
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    exit(1)

# Prepare data
features = ["Open", "High", "Low", "Close", "Volume"]
X = data[features].values
y = data["Target"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=False,  # No shuffle for time series
)

print(f"   ✓ Train set: {X_train.shape[0]} samples")
print(f"   ✓ Test set: {X_test.shape[0]} samples")

# Test 1: Logistic Regression
print("\n2. Testing Logistic Regression Model...")
start = time.time()
try:
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"   ✓ Training time: {time.time() - start:.2f}s")
    print(f"   ✓ Accuracy: {lr_acc:.4f} ({lr_acc * 100:.2f}%)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Neural Network (MLP)
print("\n3. Testing Neural Network (MLP) Model...")
start = time.time()
try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100, 50), max_iter=200, random_state=42, early_stopping=True
    )
    mlp_model.fit(X_train_scaled, y_train)
    mlp_pred = mlp_model.predict(X_test_scaled)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    print(f"   ✓ Training time: {time.time() - start:.2f}s")
    print(f"   ✓ Accuracy: {mlp_acc:.4f} ({mlp_acc * 100:.2f}%)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Enhanced Neural Network with Features
print("\n4. Testing Enhanced Neural Network...")
start = time.time()
try:
    # Add technical indicators
    data_enhanced = data.copy()
    data_enhanced["SMA_5"] = data_enhanced["Close"].rolling(window=5).mean()
    data_enhanced["SMA_20"] = data_enhanced["Close"].rolling(window=20).mean()
    data_enhanced["High_Low_Ratio"] = data_enhanced["High"] / data_enhanced["Low"]
    data_enhanced["Close_Open_Ratio"] = data_enhanced["Close"] / data_enhanced["Open"]
    data_enhanced["Volatility"] = data_enhanced["Close"].rolling(window=5).std()
    data_enhanced.dropna(inplace=True)

    enhanced_features = features + [
        "SMA_5",
        "SMA_20",
        "High_Low_Ratio",
        "Close_Open_Ratio",
        "Volatility",
    ]
    X_enh = data_enhanced[enhanced_features].values
    y_enh = data_enhanced["Target"].values

    X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(
        X_enh, y_enh, test_size=0.2, random_state=42, shuffle=False
    )

    scaler_enh = StandardScaler()
    X_train_enh_scaled = scaler_enh.fit_transform(X_train_enh)
    X_test_enh_scaled = scaler_enh.transform(X_test_enh)

    mlp_enh = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        max_iter=200,
        random_state=42,
        early_stopping=True,
    )
    mlp_enh.fit(X_train_enh_scaled, y_train_enh)
    enh_pred = mlp_enh.predict(X_test_enh_scaled)
    enh_acc = accuracy_score(y_test_enh, enh_pred)
    print(f"   ✓ Training time: {time.time() - start:.2f}s")
    print(f"   ✓ Features used: {len(enhanced_features)}")
    print(f"   ✓ Accuracy: {enh_acc:.4f} ({enh_acc * 100:.2f}%)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY - Model Comparison")
print("=" * 70)
print(f"{'Model':<30} {'Accuracy':<15} {'Performance':<20}")
print("-" * 70)
print(
    f"{'Logistic Regression':<30} {lr_acc:.4f} ({lr_acc * 100:.2f}%)  {'⭐' if lr_acc > 0.55 else '○'}"
)
print(
    f"{'Neural Network (Basic)':<30} {mlp_acc:.4f} ({mlp_acc * 100:.2f}%)  {'⭐' if mlp_acc > 0.55 else '○'}"
)
print(
    f"{'Neural Network (Enhanced)':<30} {enh_acc:.4f} ({enh_acc * 100:.2f}%)  {'⭐' if enh_acc > 0.55 else '○'}"
)
print("=" * 70)

# Best model
best_model = max(
    [
        ("Logistic Regression", lr_acc),
        ("Neural Network (Basic)", mlp_acc),
        ("Neural Network (Enhanced)", enh_acc),
    ],
    key=lambda x: x[1],
)

print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1] * 100:.2f}% accuracy")

# Trading simulation
print("\n" + "=" * 70)
print("5. Simple Trading Simulation (Last 100 days)")
print("=" * 70)

initial_balance = 10000
balance = initial_balance
shares = 0
trades = 0

# Use enhanced model for predictions
recent_data = data_enhanced.iloc[-100:]
X_recent = scaler_enh.transform(recent_data[enhanced_features].values)
predictions = mlp_enh.predict(X_recent)

for i, (idx, row) in enumerate(recent_data.iterrows()):
    price = row["Close"]
    pred = predictions[i]

    # Simple strategy: Buy if predict UP (1), Sell if predict DOWN (0)
    if pred == 1 and shares == 0 and balance >= price:
        # Buy 1 share
        shares = 1
        balance -= price
        trades += 1
    elif pred == 0 and shares > 0:
        # Sell all shares
        balance += shares * price
        shares = 0
        trades += 1

# Final value
final_price = recent_data.iloc[-1]["Close"]
final_value = balance + (shares * final_price)
profit = final_value - initial_balance
profit_pct = (profit / initial_balance) * 100

print(f"   Initial Balance: ${initial_balance:,.2f}")
print(f"   Final Balance: ${balance:,.2f}")
print(f"   Shares Held: {shares}")
print(f"   Final Value: ${final_value:,.2f}")
print(f"   Profit/Loss: ${profit:,.2f} ({profit_pct:+.2f}%)")
print(f"   Total Trades: {trades}")

# Buy and hold comparison
buy_hold_shares = initial_balance / recent_data.iloc[0]["Close"]
buy_hold_value = buy_hold_shares * final_price
buy_hold_profit = buy_hold_value - initial_balance
buy_hold_pct = (buy_hold_profit / initial_balance) * 100

print(f"\n   Buy & Hold Strategy:")
print(f"   Final Value: ${buy_hold_value:,.2f}")
print(f"   Profit/Loss: ${buy_hold_profit:,.2f} ({buy_hold_pct:+.2f}%)")

if profit > buy_hold_profit:
    print(
        f"\n   ✓ AI Strategy outperformed Buy & Hold by {profit - buy_hold_profit:,.2f}!"
    )
else:
    print(
        f"\n   ○ Buy & Hold outperformed AI Strategy by {buy_hold_profit - profit:,.2f}"
    )

print("\n" + "=" * 70)
print("✓ All tests completed successfully!")
print("=" * 70)
