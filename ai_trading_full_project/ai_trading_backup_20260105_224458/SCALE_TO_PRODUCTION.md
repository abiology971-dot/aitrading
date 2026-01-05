# 🚀 Scale to Production - Practical Implementation Guide

## Transform Your Dashboard into a World-Class AI Trading Platform

---

## 📋 Table of Contents

1. [Quick Wins (Week 1-2)](#quick-wins)
2. [Data Infrastructure (Month 1)](#data-infrastructure)
3. [Advanced AI Models (Month 2-3)](#advanced-ai-models)
4. [Production Architecture (Month 4-6)](#production-architecture)
5. [Enterprise Features (Month 7-12)](#enterprise-features)
6. [Go-to-Market Strategy](#go-to-market)

---

## ⚡ Quick Wins (Week 1-2)

### **1. Add More Data Sources**

```python
# enhanced_data_fetcher.py

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import asyncio
import aiohttp

class MultiSourceDataFetcher:
    """Fetch data from multiple sources"""
    
    def __init__(self):
        self.sources = {
            'alpha_vantage': 'YOUR_API_KEY',
            'polygon': 'YOUR_API_KEY',
            'finnhub': 'YOUR_API_KEY',
        }
    
    async def fetch_yahoo(self, ticker, start, end):
        """Yahoo Finance (Free)"""
        data = yf.download(ticker, start=start, end=end, progress=False)
        return data
    
    async def fetch_alpha_vantage(self, ticker):
        """Alpha Vantage (500 calls/day free)"""
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'apikey': self.sources['alpha_vantage'],
            'outputsize': 'full'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return pd.DataFrame(data['Time Series (Daily)']).T
    
    async def fetch_polygon(self, ticker, start, end):
        """Polygon.io (Good for real-time data)"""
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        params = {'apiKey': self.sources['polygon']}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return pd.DataFrame(data['results'])
    
    async def fetch_all(self, ticker, start, end):
        """Fetch from all sources simultaneously"""
        tasks = [
            self.fetch_yahoo(ticker, start, end),
            self.fetch_alpha_vantage(ticker),
            self.fetch_polygon(ticker, start, end),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and deduplicate
        valid_results = [r for r in results if not isinstance(r, Exception)]
        if valid_results:
            return pd.concat(valid_results).drop_duplicates()
        return None

# Usage
async def main():
    fetcher = MultiSourceDataFetcher()
    data = await fetcher.fetch_all('AAPL', '2023-01-01', '2024-01-01')
    print(f"Fetched {len(data)} rows from multiple sources")

if __name__ == '__main__':
    asyncio.run(main())
```

### **2. Add Cryptocurrency Support**

```python
# crypto_data_fetcher.py

import ccxt
import pandas as pd
from datetime import datetime, timedelta

class CryptoDataFetcher:
    """Fetch crypto data from major exchanges"""
    
    def __init__(self):
        self.exchanges = {
            'binance': ccxt.binance(),
            'coinbase': ccxt.coinbasepro(),
            'kraken': ccxt.kraken(),
        }
    
    def fetch_ohlcv(self, exchange_name, symbol, timeframe='1d', limit=1000):
        """Fetch OHLCV data"""
        exchange = self.exchanges[exchange_name]
        
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching {symbol} from {exchange_name}: {e}")
            return None
    
    def fetch_all_exchanges(self, symbol='BTC/USDT', timeframe='1d'):
        """Fetch from all exchanges and compare"""
        results = {}
        for exchange_name in self.exchanges:
            data = self.fetch_ohlcv(exchange_name, symbol, timeframe)
            if data is not None:
                results[exchange_name] = data
        return results
    
    def get_arbitrage_opportunities(self, symbol='BTC/USDT'):
        """Find price differences across exchanges"""
        data = self.fetch_all_exchanges(symbol)
        
        if len(data) < 2:
            return None
        
        latest_prices = {
            exchange: df['close'].iloc[-1] 
            for exchange, df in data.items()
        }
        
        max_exchange = max(latest_prices, key=latest_prices.get)
        min_exchange = min(latest_prices, key=latest_prices.get)
        
        spread = latest_prices[max_exchange] - latest_prices[min_exchange]
        spread_pct = (spread / latest_prices[min_exchange]) * 100
        
        return {
            'symbol': symbol,
            'buy_from': min_exchange,
            'buy_price': latest_prices[min_exchange],
            'sell_to': max_exchange,
            'sell_price': latest_prices[max_exchange],
            'spread': spread,
            'spread_pct': spread_pct,
        }

# Usage
fetcher = CryptoDataFetcher()

# Get Bitcoin data
btc_data = fetcher.fetch_all_exchanges('BTC/USDT')
print(f"Fetched BTC data from {len(btc_data)} exchanges")

# Find arbitrage
arb = fetcher.get_arbitrage_opportunities('BTC/USDT')
print(f"Arbitrage opportunity: {arb['spread_pct']:.2f}%")
```

### **3. Add More Technical Indicators**

```python
# advanced_indicators.py

import pandas as pd
import numpy as np
import talib

class AdvancedIndicators:
    """Calculate 100+ technical indicators"""
    
    @staticmethod
    def add_all_indicators(df):
        """Add all available indicators"""
        
        # Price-based indicators
        df = AdvancedIndicators.add_moving_averages(df)
        df = AdvancedIndicators.add_momentum_indicators(df)
        df = AdvancedIndicators.add_volatility_indicators(df)
        df = AdvancedIndicators.add_volume_indicators(df)
        df = AdvancedIndicators.add_trend_indicators(df)
        
        return df
    
    @staticmethod
    def add_moving_averages(df):
        """Multiple moving averages"""
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            # Simple Moving Average
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            
            # Exponential Moving Average
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            
            # Weighted Moving Average
            df[f'WMA_{period}'] = talib.WMA(df['Close'], timeperiod=period)
        
        return df
    
    @staticmethod
    def add_momentum_indicators(df):
        """Momentum and oscillators"""
        
        # RSI (multiple periods)
        for period in [14, 21, 28]:
            df[f'RSI_{period}'] = talib.RSI(df['Close'], timeperiod=period)
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Stochastic
        df['STOCH_k'], df['STOCH_d'] = talib.STOCH(
            df['High'], df['Low'], df['Close']
        )
        
        # Williams %R
        df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # ROC (Rate of Change)
        for period in [10, 20, 30]:
            df[f'ROC_{period}'] = talib.ROC(df['Close'], timeperiod=period)
        
        # CCI (Commodity Channel Index)
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df):
        """Volatility measures"""
        
        # ATR (Average True Range)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            df['Close'], timeperiod=20
        )
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_pct'] = (df['Close'] - df['BB_lower']) / df['BB_width']
        
        # Keltner Channels
        ema = df['Close'].ewm(span=20).mean()
        atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=20)
        df['KC_upper'] = ema + (2 * atr)
        df['KC_lower'] = ema - (2 * atr)
        
        # Historical Volatility
        returns = np.log(df['Close'] / df['Close'].shift(1))
        df['HV_10'] = returns.rolling(10).std() * np.sqrt(252) * 100
        df['HV_20'] = returns.rolling(20).std() * np.sqrt(252) * 100
        
        return df
    
    @staticmethod
    def add_volume_indicators(df):
        """Volume-based indicators"""
        
        # On-Balance Volume
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Accumulation/Distribution
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Chaikin Money Flow
        df['CMF'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume Rate of Change
        df['VROC'] = df['Volume'].pct_change(periods=10) * 100
        
        # Volume Moving Average
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    @staticmethod
    def add_trend_indicators(df):
        """Trend identification"""
        
        # ADX (Average Directional Index)
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Parabolic SAR
        df['SAR'] = talib.SAR(df['High'], df['Low'])
        
        # Aroon
        df['AROON_up'], df['AROON_down'] = talib.AROON(
            df['High'], df['Low'], timeperiod=25
        )
        
        # Supertrend
        df = AdvancedIndicators.calculate_supertrend(df)
        
        return df
    
    @staticmethod
    def calculate_supertrend(df, period=10, multiplier=3):
        """Custom Supertrend indicator"""
        hl2 = (df['High'] + df['Low']) / 2
        atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(period, len(df)):
            if df['Close'].iloc[i] > upper_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif df['Close'].iloc[i] < lower_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
        
        df['Supertrend'] = supertrend
        df['Supertrend_direction'] = direction
        
        return df

# Usage
import yfinance as yf

# Download data
data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')

# Add all indicators
indicators = AdvancedIndicators()
data_with_indicators = indicators.add_all_indicators(data)

print(f"Added {len(data_with_indicators.columns)} features")
print(data_with_indicators.tail())
```

---

## 🗄️ Data Infrastructure (Month 1)

### **1. Set Up PostgreSQL + TimescaleDB**

```bash
# Install TimescaleDB (Time-series optimized PostgreSQL)

# Using Docker
docker run -d --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  timescale/timescaledb:latest-pg14

# Connect and create database
psql -U postgres -h localhost
CREATE DATABASE trading_data;
\c trading_data
CREATE EXTENSION IF NOT EXISTS timescaledb;

# Create hypertable for OHLCV data
CREATE TABLE ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    exchange TEXT
);

SELECT create_hypertable('ohlcv', 'time');

# Create indexes
CREATE INDEX ON ohlcv (symbol, time DESC);
CREATE INDEX ON ohlcv (exchange, symbol, time DESC);

# Enable compression (save 90% storage)
ALTER TABLE ohlcv SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'symbol, exchange'
);

SELECT add_compression_policy('ohlcv', INTERVAL '7 days');
```

```python
# database_manager.py

import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
from contextlib import contextmanager

class TradingDatabase:
    """Manage trading data in TimescaleDB"""
    
    def __init__(self, host='localhost', database='trading_data', 
                 user='postgres', password='password'):
        self.conn_params = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
        }
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(**self.conn_params)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def insert_ohlcv(self, df, symbol, exchange='yahoo'):
        """Insert OHLCV data efficiently"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare data
            data = [
                (row.Index, symbol, row.Open, row.High, row.Low, 
                 row.Close, row.Volume, exchange)
                for row in df.itertuples()
            ]
            
            # Batch insert (10x faster than row-by-row)
            query = """
                INSERT INTO ohlcv (time, symbol, open, high, low, close, volume, exchange)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """
            execute_batch(cursor, query, data, page_size=1000)
            
            print(f"Inserted {len(data)} rows for {symbol}")
    
    def get_latest_data(self, symbol, limit=1000):
        """Get latest data for a symbol"""
        query = """
            SELECT time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s
            ORDER BY time DESC
            LIMIT %s
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql(query, conn, params=(symbol, limit))
            df.set_index('time', inplace=True)
            return df
    
    def get_data_range(self, symbol, start_date, end_date):
        """Get data for a date range"""
        query = """
            SELECT time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s AND time BETWEEN %s AND %s
            ORDER BY time
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
            df.set_index('time', inplace=True)
            return df
    
    def get_multiple_symbols(self, symbols, start_date, end_date):
        """Get data for multiple symbols (parallel)"""
        query = """
            SELECT time, symbol, close
            FROM ohlcv
            WHERE symbol = ANY(%s) AND time BETWEEN %s AND %s
            ORDER BY time
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql(query, conn, params=(symbols, start_date, end_date))
            # Pivot to wide format
            df_pivot = df.pivot(index='time', columns='symbol', values='close')
            return df_pivot

# Usage
db = TradingDatabase()

# Insert data
import yfinance as yf
aapl = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
db.insert_ohlcv(aapl, 'AAPL')

# Retrieve data
latest = db.get_latest_data('AAPL', limit=100)
print(latest.head())
```

### **2. Set Up Redis for Caching**

```python
# redis_cache.py

import redis
import json
import pandas as pd
from functools import wraps
import hashlib

class RedisCache:
    """Redis caching layer for faster data access"""
    
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, decode_responses=True
        )
    
    def cache_decorator(self, ttl=3600):
        """Decorator to cache function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key = self._generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached = self.redis_client.get(key)
                if cached:
                    print(f"Cache hit for {key}")
                    return json.loads(cached)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Store in cache
                self.redis_client.setex(
                    key, ttl, json.dumps(result, default=str)
                )
                print(f"Cached result for {key}")
                
                return result
            return wrapper
        return decorator
    
    def _generate_key(self, func_name, args, kwargs):
        """Generate cache key from function name and arguments"""
        key_data = f"{func_name}:{args}:{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cache_dataframe(self, key, df, ttl=3600):
        """Cache pandas DataFrame"""
        # Convert to JSON
        json_data = df.to_json(orient='split', date_format='iso')
        self.redis_client.setex(key, ttl, json_data)
    
    def get_dataframe(self, key):
        """Retrieve cached DataFrame"""
        json_data = self.redis_client.get(key)
        if json_data:
            return pd.read_json(json_data, orient='split')
        return None
    
    def invalidate_pattern(self, pattern):
        """Delete all keys matching pattern"""
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)
            print(f"Invalidated {len(keys)} cache keys")

# Usage
cache = RedisCache()

# Use as decorator
@cache.cache_decorator(ttl=3600)
def expensive_calculation(ticker, days=30):
    # Simulate expensive operation
    import time
    time.sleep(2)
    return {'ticker': ticker, 'result': 'calculated'}

# First call - takes 2 seconds
result1 = expensive_calculation('AAPL', days=30)

# Second call - instant (from cache)
result2 = expensive_calculation('AAPL', days=30)

# Cache DataFrames
import yfinance as yf
data = yf.download('AAPL', period='1mo')
cache.cache_dataframe('AAPL:1mo', data, ttl=300)

# Retrieve
cached_data = cache.get_dataframe('AAPL:1mo')
```

---

## 🤖 Advanced AI Models (Month 2-3)

### **1. Implement Ensemble Model**

```python
# ensemble_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class EnsembleTrading:
    """Ensemble of multiple ML models for better predictions"""
    
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42),
            'neural_net': MLPClassifier(hidden_layers=(100, 50), max_iter=500, random_state=42),
        }
        
        # Meta-learner (stacking)
        self.meta_learner = LogisticRegression(random_state=42)
        
        self.trained = False
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train all models and meta-learner"""
        
        print("Training individual models...")
        meta_features_train = []
        meta_features_val = []
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions for meta-learner
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]
            
            meta_features_train.append(train_pred)
            meta_features_val.append(val_pred)
            
            # Evaluate
            val_pred_class = model.predict(X_val)
            acc = accuracy_score(y_val, val_pred_class)
            print(f"    {name} accuracy: {acc:.4f}")
        
        # Train meta-learner
        print("\nTraining meta-learner (stacking)...")
        X_meta_train = np.column_stack(meta_features_train)
        X_meta_val = np.column_stack(meta_features_val)
        
        self.meta_learner.fit(X_meta_train, y_train)
        
        # Final ensemble prediction
        meta_pred = self.meta_learner.predict(X_meta_val)
        final_acc = accuracy_score(y_val, meta_pred)
        
        print(f"\nEnsemble accuracy: {final_acc:.4f}")
        print(f"Improvement: {final_acc - max([accuracy_score(y_val, model.predict(X_val)) for model in self.models.values()]):.4f}")
        
        self.trained = True
        
        return {
            'individual_accuracies': {
                name: accuracy_score(y_val, model.predict(X_val))
                for name, model in self.models.items()
            },
            'ensemble_accuracy': final_acc
        }
    
    def predict(self, X):
        """Predict using ensemble"""
        if not self.trained:
            raise ValueError("Models not trained yet!")
        
        # Get predictions from all models
        predictions = []
        for model in self.models.values():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        # Stack predictions
        X_meta = np.column_stack(predictions)
        
        # Final prediction
        final_pred = self.meta_learner.predict(X_meta)
        final_proba = self.meta_learner.predict_proba(X_meta)
        
        return final_pred, final_proba
    
    def predict_with_confidence(self, X):
        """Predict with confidence from individual models"""
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X)
            proba = model.predict_proba(X)
            predictions[name] = {
                'prediction': pred[0],
                'confidence': max(proba[0])
            }
        
        # Ensemble prediction
        ensemble_pred, ensemble_proba = self.predict(X)
        
        predictions['ensemble'] = {
            'prediction': ensemble_pred[0],
            'confidence': max(ensemble_proba[0])
        }
        
        return predictions

# Usage
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('stock_data.csv')
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features].values
y = data['Target'].values

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=False
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train ensemble
ensemble = EnsembleTrading()
results = ensemble.train(X_train_scaled, y_train, X_val_scaled, y_val)

# Test ensemble
test_pred, test_proba = ensemble.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# Get prediction with confidence for single sample
sample = X_test_scaled[0:1]
detailed_pred = ensemble.predict_with_confidence(sample)

print("\nDetailed predictions:")
for model_name, pred_info in detailed_pred.items():
    print(f"{model_name}: {pred_info['prediction']} (confidence: {pred_info['confidence']:.2%})")
```

### **2. Add Sentiment Analysis**

```python
# sentiment_analysis.py

import requests
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
import tweepy
from newsapi import NewsApiClient

class SentimentAnalyzer:
    """Analyze market sentiment from news and social media"""
    
    def __init__(self, news_api_key=None, twitter_keys=None):
        self.news_api = NewsApiClient(api_key=news_api_key) if news_api_key else None
        
        if twitter_keys:
            auth = tweepy.OAuthHandler(
                twitter_keys['consumer_key'],
                twitter_keys['consumer_secret']
            )
            auth.set_access_token(
                twitter_keys['access_token'],
                twitter_keys['access_token_secret']
            )
            self.twitter_api = tweepy.API(auth)
        else:
            self.twitter_api = None
    
    def get_news_sentiment(self, ticker, days_back=7):
        """Get sentiment from news articles"""
        if not self.news_api:
            return None
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        # Get news articles
        articles = self.news_api.get_everything(
            q=ticker,
            from_param=from_date.strftime('%Y-%m-%d'),
            to=to_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy'
        )
        
        if not articles['articles']:
            return {'sentiment': 0, 'count': 0}
        
        # Analyze sentiment
        sentiments = []
        for article in articles['articles']:
            text = f"{article['title']} {article.get('description', '')}"
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
        
        return {
            'sentiment': sum(sentiments) / len(sentiments),
            'count': len(sentiments),
            'articles': articles['articles'][:5]  # Top 5 articles
        }
    
    def get_twitter_sentiment(self, ticker, count=100):
        """Get sentiment from Twitter"""
        if not self.twitter_api:
            return None
        
        # Search tweets
        tweets = self.twitter_api.search_tweets(
            q=f"${ticker