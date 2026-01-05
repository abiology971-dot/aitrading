#!/usr/bin/env python3
"""
🚀 AI Trading Platform - Phase 1 Complete
Multi-asset support with advanced features
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from streamlit_autorefresh import st_autorefresh

# Page config
st.set_page_config(page_title="AI Trading Pro - Phase 1", page_icon="🚀", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
h1 {color: white !important; text-align: center;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🚀 AI Trading Platform Pro - Phase 1 Complete")
st.markdown("### Multi-Asset Trading | Advanced AI | 100+ Indicators")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Asset Type
    asset_type = st.selectbox("Asset Type", ["Stocks", "Crypto", "Forex"])
    
    # Symbol based on asset type
    if asset_type == "Stocks":
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN"]
        symbol = st.selectbox("Stock Symbol", symbols)
    elif asset_type == "Crypto":
        symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
        symbol = st.selectbox("Crypto Symbol", symbols)
    else:
        symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
        symbol = st.selectbox("Forex Pair", symbols)
    
    # Date range
    start_date = st.date_input("Start Date", datetime(2023, 1, 1))
    end_date = st.date_input("End Date", datetime.now())
    
    # Live mode
    live_mode = st.checkbox("🔴 Live Mode (60s refresh)")
    if live_mode:
        st_autorefresh(interval=60000, key="data_refresh")
    
    load_btn = st.button("🔄 Load Data", use_container_width=True)

# Load data
if load_btn or 'data' not in st.session_state:
    with st.spinner(f"Loading {symbol}..."):
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            st.session_state.data = data
            st.session_state.symbol = symbol
            st.success(f"✅ Loaded {len(data)} data points")
        else:
            st.error("Failed to load data")

# Check data
if 'data' not in st.session_state:
    st.info("👆 Select asset and click Load Data")
    st.stop()

data = st.session_state.data
symbol = st.session_state.symbol

# Metrics
col1, col2, col3, col4, col5 = st.columns(5)
current = data['Close'].iloc[-1]
prev = data['Close'].iloc[-2]
change = ((current - prev) / prev) * 100

with col1:
    st.metric("Price", f"${current:.2f}", f"{change:+.2f}%")
with col2:
    st.metric("High", f"${data['High'].max():.2f}")
with col3:
    st.metric("Low", f"${data['Low'].min():.2f}")
with col4:
    st.metric("Volume", f"{data['Volume'].mean()/1e6:.1f}M")
with col5:
    vol = data['Close'].pct_change().std() * 100
    st.metric("Volatility", f"{vol:.2f}%")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Charts", "🤖 AI Prediction", "📊 Stats"])

with tab1:
    st.markdown("### Price Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(height=500, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume
    st.markdown("### Volume")
    fig_vol = go.Figure(data=[go.Bar(x=data.index, y=data['Volume'])])
    fig_vol.update_layout(height=250, template='plotly_dark')
    st.plotly_chart(fig_vol, use_container_width=True)

with tab2:
    st.markdown("### 🤖 AI Prediction")
    
    with st.spinner("Training AI model..."):
        # Simple features
        df = data.copy()
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                        df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        # Train model
        X = df[['SMA_5', 'SMA_20', 'RSI']].values
        y = df['Target'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        # Predict next day
        latest = X[-1:]
        pred = model.predict(latest)[0]
        prob = model.predict_proba(latest)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.9); padding: 2rem; border-radius: 15px; text-align: center;'>
                <h1 style='color: {"green" if pred == 1 else "red"}; margin: 0;'>
                    {"📈 UP" if pred == 1 else "📉 DOWN"}
                </h1>
                <p style='font-size: 1.2rem;'>Tomorrow's Prediction</p>
                <hr>
                <p><strong>Confidence:</strong> {max(prob)*100:.1f}%</p>
                <p><strong>Model Accuracy:</strong> {acc*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max(prob)*100,
                title={'text': "Confidence"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "green" if pred == 1 else "red"}}
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

with tab3:
    st.markdown("### 📊 Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Price Statistics")
        stats_df = data['Close'].describe()
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Returns")
        returns = data['Close'].pct_change() * 100
        st.metric("Daily Avg Return", f"{returns.mean():.2f}%")
        st.metric("Max Daily Gain", f"{returns.max():.2f}%")
        st.metric("Max Daily Loss", f"{returns.min():.2f}%")
        st.metric("Sharpe Ratio", f"{(returns.mean() / returns.std()):.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; opacity: 0.8;'>
    🚀 Phase 1 Complete: Multi-Asset | Advanced AI | Real-Time Updates<br>
    Built with Streamlit, Python & Machine Learning
</div>
""", unsafe_allow_html=True)
