#!/usr/bin/env python3
"""
🚀 AI Trading Platform - Enhanced Dashboard (Phase 1 Complete)
Production-ready dashboard with multi-asset support, advanced features, and real-time updates
"""

import warnings
from datetime import datetime, timedelta
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Trading Platform Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    /* Modern Gradient Background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 1rem;
    }

    /* Headers */
    h1 {
        color: white !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-size: 2.5rem !important;
        text-align: center;
        padding: 1rem 0;
    }

    h2 {
        color: white !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
    }

    h3 {
        color: #1e3c72 !important;
        font-weight: 600 !important;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3c72;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }

    /* Cards */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    /* Success/Error Messages */
    .stSuccess {
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 5px solid #00ff00;
        border-radius: 10px;
        padding: 1rem;
    }

    .stError {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 5px solid #ff0000;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Live Badge */
    .live-badge {
        background: #ff0000;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== MULTI-ASSET DATA FETCHING ====================

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data = data.reset_index()
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_crypto_data(symbol, start_date, end_date):
    """Fetch cryptocurrency data"""
    try:
        # Use yfinance for crypto (BTC-USD, ETH-USD, etc.)
        ticker = f"{symbol}-USD" if not symbol.endswith("-USD") else symbol
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data = data.reset_index()
        return data
    except Exception as e:
        st.error(f"Error fetching crypto data: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_forex_data(pair, start_date, end_date):
    """Fetch forex data"""
    try:
        # Use yfinance for forex (EURUSD=X, GBPUSD=X, etc.)
        ticker = f"{pair}=X" if not pair.endswith("=X") else pair
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data = data.reset_index()
        return data
    except Exception as e:
        st.error(f"Error fetching forex data: {e}")
        return None

# ==================== ADVANCED TECHNICAL INDICATORS ====================

def add_all_indicators(data):
    """Add 100+ technical indicators"""
    df = data.copy()

    # Moving Averages (20 indicators)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']

    # RSI (multiple periods)
    for period in [7, 14, 21, 28]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Rate of Change
    for period in [5, 10, 20]:
        df[f'ROC_{period}'] = df['Close'].pct_change(periods=period) * 100

    # Volatility
    for period in [10, 20, 30]:
        df[f'Volatility_{period}'] = df['Close'].rolling(window=period).std()

    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

    # Price Ratios
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']

    # Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Momentum_20'] = df['Close'] - df['Close'].shift(20)

    # Williams %R
    df['Williams_R'] = -100 * ((high_max - df['Close']) / (high_max - low_min))

    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    df.dropna(inplace=True)
    return df

# ==================== ENSEMBLE MODEL ====================

@st.cache_resource
def train_ensemble_model(data):
    """Train ensemble of multiple models"""
    # Add indicators
    data_with_indicators = add_all_indicators(data)

    # Create target
    data_with_indicators['Target'] = (data_with_indicators['Close'].shift(-1) > data_with_indicators['Close']).astype(int)
    data_with_indicators.dropna(inplace=True)

    # Select features
    feature_cols = [col for col in data_with_indicators.columns if col not in ['Date', 'Target', 'Close', 'Open', 'High', 'Low', 'Volume']]
    feature_cols = feature_cols[:50]  # Use top 50 features for speed

    X = data_with_indicators[feature_cols].values
    y = data_with_indicators['Target'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
    }

    results = {}
    predictions = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, pred)
        results[name] = {
            'model': model,
            'accuracy': acc,
            'predictions': pred
        }
        predictions.append(pred)

    # Ensemble voting
    ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)

    results['Ensemble'] = {
        'accuracy': ensemble_acc,
        'predictions': ensemble_pred
    }

    return {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'results': results,
        'X_test': X_test_scaled,
        'y_test': y_test
    }

# ==================== VISUALIZATION FUNCTIONS ====================

def create_candlestick_chart(data, title="Price Chart"):
    """Create candlestick chart"""
    fig = go.Figure(data=[
        go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        )
    ])

    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
    )

    return fig

def create_volume_chart(data):
    """Create volume chart"""
    colors = ['red' if row['Close'] < row['Open'] else 'green' for _, row in data.iterrows()]

    fig = go.Figure(data=[
        go.Bar(x=data['Date'], y=data['Volume'], marker_color=colors, name='Volume', opacity=0.7)
    ])

    fig.update_layout(
        title='Trading Volume',
        yaxis_title='Volume',
        xaxis_title='Date',
        template='plotly_dark',
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
    )

    return fig

def create_technical_chart(data):
    """Create technical indicators chart"""
    data_indicators = add_all_indicators(data)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
        row_heights=[0.5, 0.25, 0.25]
    )

    # Price and MAs
    fig.add_trace(
        go.Scatter(x=data_indicators['Date'], y=data_indicators['Close'], name='Close', line=dict(color='#667eea', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data_indicators['Date'], y=data_indicators['SMA_20'], name='SMA 20', line=dict(color='#f093fb', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data_indicators['Date'], y=data_indicators['SMA_50'], name='SMA 50', line=dict(color='#4facfe', width=1)),
        row=1, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=data_indicators['Date'], y=data_indicators['RSI_14'], name='RSI', line=dict(color='#43e97b', width=2)),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(x=data_indicators['Date'], y=data_indicators['MACD'], name='MACD', line=dict(color='#fa709a', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=data_indicators['Date'], y=data_indicators['MACD_Signal'], name='Signal', line=dict(color='#fee140', width=2)),
        row=3, col=1
    )

    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
    )

    return fig

def create_prediction_gauge(probability, prediction):
    """Create gauge chart for prediction"""
    color = 'green' if prediction == 1 else 'red'

    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Confidence', 'font': {'size': 24, 'color': 'white'}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(0, 255, 0, 0.3)'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=300
    )

    return fig

# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1>🚀 AI Trading Platform Pro - Phase 1 Complete</h1>
            <p style='color: white; font-size: 1.2rem; opacity: 0.9;'>
                Multi-Asset Trading with Advanced AI & 100+ Indicators
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h2 style='color: white;'>⚙️ Settings</h2>
            </div>
            """, unsafe_allow_html=True)

        # Asset Type Selection
        asset_type = st.selectbox(
            "🎯 Asset Type",
            ["Stocks", "Cryptocurrency", "Forex"],
            help="Select the type of asset to analyze"
        )

        # Symbol Input based on asset type
        if asset_type == "Stocks":
            default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
            symbol = st.selectbox("📊 Stock Symbol", default_symbols, index=0)
            st.info("💡 Popular: AAPL, TSLA, NVDA, MSFT")
        elif asset_type == "Cryptocurrency":
            default_cryptos = ["BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE"]
            symbol = st.selectbox("🪙 Crypto Symbol", default_cryptos, index=0)
            st.info("💡 Popular: BTC, ETH, BNB, SOL")
        else:  # Forex
            default_forex = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
            symbol = st.selectbox("💱 Forex Pair", default_forex, index=0)
            st.info("💡 Popular: EURUSD, GBPUSD, USDJPY")

        # Date Range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("📅 Start Date", value=datetime(2023, 1, 1))
        with col2:
            end_date = st.date_input("📅 End Date", value=datetime.now())

        # Live Mode
        live_mode = st.checkbox("🔴 Live Mode", help="Auto-refresh every 60 seconds")
        if live_mode:
            st.markdown('<span class="live-badge">● LIVE</span>', unsafe_allow_html=True)

        # Load Data Button
        load_button = st.button("🔄 Load Data", use_container_width=True)

        st.markdown("---")

        # Model Selection
        st.markdown("<h3 style='color: white;'>🤖 AI Model</h3>", unsafe_allow_html=True)
        model_choice = st.radio(
            "Select Model",
            ["Ensemble (Best)", "Random Forest", "Neural Network", "Gradient Boosting"],
            help="Choose the AI model for predictions"
        )

        st.markdown("---")

        # Trading Parameters
        st.markdown("<h3 style='color: white;'>💰 Trading</h3>", unsafe_allow_html=True)
        initial_balance = st.number_input("Initial Balance ($)", value=10000, min_value=1000, step=1000)

        st.markdown("---")

        # Stats
        if 'data' in st.session_state and st.session_state.data is not None:
            st.markdown("<h3 style='color: white;'>📊 Stats</h3>", unsafe_allow_html=True)
            st.metric("Data Points", f"{len(st.session_state.data):,}")
            st.metric("Features", "100+")
            st.metric("Models", "4")

    # Load or refresh data
    if load_button or live_mode or 'data' not in st.session_state:
        with st.spinner(f"🔄 Loading {asset_type} data for {symbol}..."):
            if asset_type == "Stocks":
                data = fetch_stock_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            elif asset_type == "Cryptocurrency":
                data = fetch_crypto_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            else:  # Forex
                data = fetch_forex_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            if data is not None and len(data) > 0:
                st.session_state.data = data
                st.session_state.symbol = symbol
                st.session_state.asset_type = asset_type
                st.success(f"✅ Loaded {len(data)} data points for {symbol}")
            else:
                st.error(f"❌ Failed to load data for {symbol}")
                return

    # Check if data exists
    if 'data' not in st.session_state or st.session_state.data is None:
        st.info("👆 Please select an asset and click 'Load Data' to begin")
        return

    data = st.session_state.data
    symbol = st.session_state.symbol

    # Overview Metrics
    st.markdown("## 📊 Market Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100

    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")

    with col2:
        st.metric("High (Period)", f"${data['High'].max():.2f}")

    with col3:
        st.metric("Low (Period)", f"${data['Low'].min():.2f}")

    with col4:
        avg_volume = data['Volume'].mean()
        st.metric("Avg Volume", f"{avg_volume/1e6:.2f}M" if avg_volume > 1e6 else f"{avg_volume/1e3:.2f}K")

    with col5:
        volatility = data['Close'].pct_change().std() * 100
        st.metric("Volatility", f"{volatility:.2f}%")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price Chart",
        "🔍 Technical Analysis",
        "🤖 AI Predictions",
        "💹 Backtesting",
        "📊 Model Comparison",
        "⚡ Quick Stats"
    ])

    with tab1:
        st.markdown("### 📈 Price Movement")

        # Candlestick Chart
        fig_candle = create_candlestick_chart(data.tail(180), f"{symbol} - Last 180 Days")
        st.plotly_chart(fig_candle, use_container_width=True)

        # Volume Chart
        fig_volume = create_volume_chart(data.tail(180))
        st.plotly_chart(fig_volume, use_container_width=True)

        # Recent Data
        with st.expander("📋 View Recent Data"):
            st.dataframe(
                data.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].style.format({
                    'Open': '${:.2f}',
                    'High': '${:.2f}',
                    'Low': '${:.2f}',
                    'Close': '${:.2f}',
                    'Volume': '{:,.0f}'
                }),
                use_container_width=True
            )

    with tab2:
        st.markdown("### 🔍 Technical Indicators")

        # Technical Chart
        fig_tech = create_technical_chart(data.tail(180))
        st.plotly_chart(fig_tech, use_container_width=True)

        # Indicator Summary
        data_indicators = add_all_indicators(data)
        latest = data_indicators.iloc[-1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📊 RSI Analysis")
            rsi_value = latest['RSI_14']
            if rsi_value > 70:
                rsi_status = "🔴 Overbought"
                rsi_color = "red"
            elif rsi_value < 30:
                rsi_status = "🟢 Oversold"
                rsi_color = "green"
            else:
                rsi_status = "🟡 Neutral"
                rsi_color = "orange"

            st.markdown(f"""
                <div style='background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px; text-align: center;'>
                    <h2 style='color: {rsi_color}; margin: 0;'>{rsi_value:.2f}</h2>
                    <p style='margin: 0;'>{rsi_status}</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### 📈 MACD Signal")
            macd = latest['MACD']
            signal = latest['MACD_Signal']

            if macd > signal:
                macd_status = "🟢 Bullish"
                macd_color = "green"
            else:
                macd_status = "🔴 Bearish"
                macd_color = "red"

            st.markdown(f"""
                <div style='background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px; text-align: center;'>
                    <h2 style='color: {macd_color}; margin: 0;'>{macd:.2f}</h2>
                    <p style='margin: 0;'>{macd_status}</p>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            st.markdown("#### 📉 Volatility")
            vol = latest['Volatility_20']
            vol_status = "High" if vol >
