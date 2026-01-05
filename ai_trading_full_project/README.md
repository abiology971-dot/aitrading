# 🤖 AI Trading Full Project

A comprehensive machine learning-based stock trading prediction system using multiple AI approaches including Logistic Regression, Neural Networks, and Reinforcement Learning.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements various machine learning algorithms to predict stock price movements (UP/DOWN) for the next trading day. It includes:

- **Data Collection**: Automatic stock data fetching using yfinance
- **Multiple ML Models**: Logistic Regression, Neural Networks, LSTM alternative
- **Reinforcement Learning**: PPO-based trading bot
- **Feature Engineering**: Technical indicators (SMA, volatility, ratios)
- **Backtesting**: Trading simulation and performance evaluation

**Dataset**: Apple Inc. (AAPL) stock data from 2015-2024

---

## ✨ Features

- ✅ **Automated Data Pipeline**: Download and preprocess stock data
- ✅ **Multiple ML Models**: Compare different algorithms
- ✅ **Technical Indicators**: SMA, volatility, price ratios
- ✅ **Fast Testing Suite**: Test all models in under 1 minute
- ✅ **Trading Simulation**: Backtest strategies on historical data
- ✅ **Performance Metrics**: Accuracy, confusion matrix, ROI
- ✅ **RL Trading Bot**: Reinforcement learning agent for trading

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project
```bash
cd ai_trading_full_project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- numpy
- pandas
- scikit-learn
- tensorflow (optional - has compatibility issues on some systems)
- yfinance
- gym
- stable-baselines3

---

## ⚡ Quick Start

### Option 1: Fast Test (RECOMMENDED) ⭐
Test all models in under 1 minute:
```bash
python test_all_models.py
```

This will:
- Load stock data
- Train 3 different models
- Compare performance
- Run trading simulation
- Show comprehensive results

### Option 2: Step-by-Step

**1. Download Stock Data**
```bash
python fetch_data.py
```

**2. Train Logistic Regression (Fast)**
```bash
python logistic_model.py
```

**3. Train Neural Network (Recommended)**
```bash
python lstm_alternative.py
```

**4. Train RL Bot (Slow - Optional)**
```bash
python rl_trading_bot.py
```

---

## 📁 Project Structure

```
ai_trading_full_project/
│
├── fetch_data.py              # Download and clean stock data
├── logistic_model.py          # Basic logistic regression model
├── lstm_model.py              # LSTM model (TensorFlow - has issues)
├── lstm_alternative.py        # Alternative neural network (scikit-learn)
├── rl_trading_bot.py          # Reinforcement learning trading bot
├── test_all_models.py         # Quick test suite for all models
│
├── stock_data.csv             # Downloaded stock data (generated)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── DEBUGGING_SUMMARY.md       # Detailed debugging documentation
```

---

## 🧠 Models

### 1. Logistic Regression
- **Type**: Traditional ML
- **Speed**: Very Fast (~0.03s)
- **Accuracy**: ~53%
- **Use Case**: Baseline model, quick predictions

### 2. Neural Network (Basic)
- **Type**: Multi-layer Perceptron
- **Speed**: Fast (~0.07s)
- **Accuracy**: ~46%
- **Use Case**: Simple non-linear patterns

### 3. Neural Network (Enhanced)
- **Type**: MLP with technical indicators
- **Speed**: Fast (~0.17s)
- **Accuracy**: ~51-53%
- **Use Case**: Feature-rich predictions
- **Features**: 10 indicators including SMA, volatility, ratios

### 4. LSTM Alternative
- **Type**: Deep Neural Network with sequences
- **Speed**: Medium (~30s)
- **Accuracy**: ~53%
- **Use Case**: Time series patterns with context

### 5. Reinforcement Learning Bot
- **Type**: PPO (Proximal Policy Optimization)
- **Speed**: Slow (5-10 minutes)
- **Use Case**: Learning optimal trading strategy
- **Actions**: Hold, Buy, Sell

---

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| Logistic Regression | 52.98% | ⚡⚡⚡ | Quick baseline |
| Neural Network (Basic) | 46.36% | ⚡⚡⚡ | Simple patterns |
| Neural Network (Enhanced) | 51.45% | ⚡⚡ | Feature-rich data |
| LSTM Alternative | 53.12% | ⚡ | Sequential patterns |

### Trading Simulation (100 days)

**AI Strategy:**
- Initial: $10,000
- Final: $10,016
- Profit: +0.16%
- Trades: 21

**Buy & Hold:**
- Initial: $10,000
- Final: $10,833
- Profit: +8.34%

**Note**: Stock prediction is inherently difficult. These results are typical for ML-based trading systems. The models perform slightly better than random guessing (50%).

---

## 💻 Usage Examples

### Example 1: Download Custom Stock Data
```python
import yfinance as yf
import pandas as pd

ticker = "TSLA"  # Change to any stock
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Flatten columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

data = data.reset_index()
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data.dropna(inplace=True)
data.to_csv("stock_data.csv", index=False)
```

### Example 2: Make a Prediction
```python
import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load("best_neural_model.pkl")
scaler = joblib.load("neural_scaler.pkl")

# Load recent data
data = pd.read_csv("stock_data.csv")
recent = data.tail(5)  # Last 5 days

# Prepare features
features = ["Open", "High", "Low", "Close", "Volume"]
X = scaler.transform(recent[features].values)

# Predict
prediction = model.predict(X[-1].reshape(1, -1))
probability = model.predict_proba(X[-1].reshape(1, -1))

print(f"Prediction: {'UP' if prediction[0] == 1 else 'DOWN'}")
print(f"Confidence: {max(probability[0]):.2%}")
```

### Example 3: Custom Trading Strategy
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("stock_data.csv")

# Train model
X = data[["Open", "High", "Low", "Close", "Volume"]].iloc[:-100]
y = data["Target"].iloc[:-100]
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Simulate trading
initial_balance = 10000
balance = initial_balance
shares = 0

test_data = data.iloc[-100:]
for idx, row in test_data.iterrows():
    X_pred = [[row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]]]
    prediction = model.predict(X_pred)[0]
    
    if prediction == 1 and shares == 0:  # Buy signal
        shares = balance // row["Close"]
        balance -= shares * row["Close"]
    elif prediction == 0 and shares > 0:  # Sell signal
        balance += shares * row["Close"]
        shares = 0

final_value = balance + shares * test_data.iloc[-1]["Close"]
profit = final_value - initial_balance
print(f"Final Value: ${final_value:.2f}")
print(f"Profit: ${profit:.2f} ({profit/initial_balance*100:.2f}%)")
```

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Issue: TensorFlow errors (mutex lock failed)
**Solution**: Use `lstm_alternative.py` instead of `lstm_model.py`
```bash
python lstm_alternative.py
```

### Issue: No file 'stock_data.csv'
**Solution**: Run data fetching first
```bash
python fetch_data.py
```

### Issue: RL training too slow
**Solution**: Use the fast test suite instead
```bash
python test_all_models.py
```

### Issue: Poor accuracy
This is expected! Stock prediction is extremely difficult:
- Markets are influenced by countless external factors
- 50-54% accuracy is actually reasonable
- Try adding more features or ensemble methods

---

## 🎓 Learning Resources

### Understanding the Models

**Logistic Regression**: 
- Linear model for binary classification
- Fast and interpretable
- Good baseline for comparison

**Neural Networks**:
- Learn non-linear patterns
- Require more data and tuning
- Better for complex relationships

**LSTM (Long Short-Term Memory)**:
- Specialized for sequential data
- Remember long-term dependencies
- Good for time series

**Reinforcement Learning**:
- Learns optimal actions through trial and error
- Balances exploration vs exploitation
- Can discover novel strategies

---

## 📈 Improving the Models

### Easy Improvements:
1. Add more technical indicators (RSI, MACD, Bollinger Bands)
2. Test on multiple stocks
3. Increase training data (more years)
4. Ensemble methods (combine multiple models)

### Advanced Improvements:
1. Sentiment analysis from news/Twitter
2. Transformer models (attention mechanism)
3. Real-time data streaming
4. Portfolio optimization
5. Risk management systems

---

## ⚠️ Disclaimer

**IMPORTANT**: This project is for educational purposes only.

- ❌ **DO NOT** use for real trading without extensive testing
- ❌ **DO NOT** risk money you cannot afford to lose
- ✅ Always backtest on historical data
- ✅ Use paper trading before live trading
- ✅ Consult financial advisors for investment decisions

**Stock market prediction is extremely difficult and results may vary significantly.**

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available for educational purposes.

---

## 🙏 Acknowledgments

- **yfinance**: Stock data API
- **scikit-learn**: Machine learning library
- **stable-baselines3**: Reinforcement learning algorithms
- **pandas/numpy**: Data manipulation

---

## 📞 Support

For issues and questions:
- Check [DEBUGGING_SUMMARY.md](DEBUGGING_SUMMARY.md) for detailed troubleshooting
- Review code comments and docstrings
- Test with `test_all_models.py` first

---

## 🎯 Quick Command Reference

```bash
# Full workflow
python fetch_data.py              # Step 1: Get data
python test_all_models.py         # Step 2: Test models (FAST)

# Individual models
python logistic_model.py          # Logistic regression
python lstm_alternative.py        # Neural network
python rl_trading_bot.py          # RL bot (slow)

# Check results
ls -lh *.pkl *.h5 *.csv          # View generated files
```

---

**Made with ❤️ for ML enthusiasts and traders**

**Status**: ✅ Working | **Last Updated**: 2024 | **Version**: 1.0