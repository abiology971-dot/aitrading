#!/usr/bin/env python3
"""
🚀 AI Trading Dashboard - Ultra Modern UI/UX
A beautiful, interactive dashboard for AI-powered stock trading predictions
"""

import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS ====================
st.markdown(
    """
    <style>
    /* Main background with gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 1rem;
    }

    [data-testid="stSidebar"] .element-container {
        color: white;
    }

    /* Card styling */
    .stApp {
        background: transparent;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3c72;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
        color: #555;
    }

    /* Custom cards */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
    }

    /* Headers */
    h1 {
        color: white !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-size: 3rem !important;
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
        color: white;
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Info boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }

    /* Dataframes */
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white !important;
        font-weight: 600;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
    }

    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .element-container {
        animation: fadeIn 0.5s ease-out;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }

    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}

    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== HELPER FUNCTIONS ====================


@st.cache_data(ttl=3600)
def load_stock_data(ticker="AAPL", start_date="2015-01-01", end_date=None):
    """Load and process stock data"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data = data.reset_index()
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        data.dropna(inplace=True)

        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def add_technical_indicators(data):
    """Add technical indicators to dataframe"""
    df = data.copy()

    # Moving Averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # Exponential Moving Average
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # Volatility
    df["Volatility"] = df["Close"].rolling(window=20).std()

    # Price Ratios
    df["High_Low_Ratio"] = df["High"] / df["Low"]
    df["Close_Open_Ratio"] = df["Close"] / df["Open"]

    # Volume indicators
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]

    df.dropna(inplace=True)
    return df


@st.cache_resource
def train_models(data):
    """Train multiple ML models"""
    features = ["Open", "High", "Low", "Close", "Volume"]
    X = data[features].values
    y = data["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    # Train Neural Network with enhanced features
    data_enhanced = add_technical_indicators(data)
    enhanced_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_5",
        "SMA_20",
        "RSI",
        "MACD",
        "Volatility",
    ]

    X_enh = data_enhanced[enhanced_features].values
    y_enh = data_enhanced["Target"].values

    X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(
        X_enh, y_enh, test_size=0.2, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enh)
    X_test_scaled = scaler.transform(X_test_enh)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        max_iter=500,
        random_state=42,
        early_stopping=True,
    )
    mlp_model.fit(X_train_scaled, y_train_enh)
    mlp_pred = mlp_model.predict(X_test_scaled)
    mlp_acc = accuracy_score(y_test_enh, mlp_pred)

    return {
        "lr_model": lr_model,
        "mlp_model": mlp_model,
        "scaler": scaler,
        "lr_acc": lr_acc,
        "mlp_acc": mlp_acc,
        "y_test_lr": y_test,
        "lr_pred": lr_pred,
        "y_test_mlp": y_test_enh,
        "mlp_pred": mlp_pred,
        "X_test_lr": X_test,
        "X_test_mlp": X_test_scaled,
    }


def create_candlestick_chart(data, title="Stock Price Chart"):
    """Create interactive candlestick chart"""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
            )
        ]
    )

    fig.update_layout(
        title=title,
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )

    return fig


def create_volume_chart(data):
    """Create volume chart"""
    colors = [
        "red" if row["Close"] < row["Open"] else "green" for _, row in data.iterrows()
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=data["Date"],
                y=data["Volume"],
                marker_color=colors,
                name="Volume",
                opacity=0.7,
            )
        ]
    )

    fig.update_layout(
        title="Trading Volume",
        yaxis_title="Volume",
        xaxis_title="Date",
        template="plotly_dark",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )

    return fig


def create_technical_indicators_chart(data):
    """Create technical indicators chart"""
    data_with_indicators = add_technical_indicators(data)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Price & Moving Averages", "RSI", "MACD"),
        row_heights=[0.5, 0.25, 0.25],
    )

    # Price and MAs
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators["Date"],
            y=data_with_indicators["Close"],
            name="Close",
            line=dict(color="#667eea", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators["Date"],
            y=data_with_indicators["SMA_20"],
            name="SMA 20",
            line=dict(color="#f093fb", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators["Date"],
            y=data_with_indicators["SMA_50"],
            name="SMA 50",
            line=dict(color="#4facfe", width=1),
        ),
        row=1,
        col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators["Date"],
            y=data_with_indicators["RSI"],
            name="RSI",
            line=dict(color="#43e97b", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators["Date"],
            y=data_with_indicators["MACD"],
            name="MACD",
            line=dict(color="#fa709a", width=2),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators["Date"],
            y=data_with_indicators["Signal_Line"],
            name="Signal",
            line=dict(color="#fee140", width=2),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=True,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )

    return fig


def create_confusion_matrix_chart(y_true, y_pred, title="Confusion Matrix"):
    """Create confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["DOWN", "UP"],
            y=["DOWN", "UP"],
            colorscale="RdYlGn",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )

    return fig


def create_prediction_gauge(probability, prediction):
    """Create gauge chart for prediction confidence"""
    color = "green" if prediction == 1 else "red"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Confidence", "font": {"size": 24, "color": "white"}},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 50], "color": "rgba(255, 0, 0, 0.3)"},
                    {"range": [50, 100], "color": "rgba(0, 255, 0, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=300,
    )

    return fig


def simulate_trading(data, predictions, initial_balance=10000):
    """Simulate trading strategy"""
    balance = initial_balance
    shares = 0
    trades = []

    for i, (idx, row) in enumerate(data.iterrows()):
        price = row["Close"]
        pred = predictions[i]

        if pred == 1 and shares == 0 and balance >= price:
            shares = balance // price
            cost = shares * price
            balance -= cost
            trades.append(
                {
                    "Date": row["Date"],
                    "Action": "BUY",
                    "Price": price,
                    "Shares": shares,
                    "Cost": cost,
                    "Balance": balance,
                }
            )
        elif pred == 0 and shares > 0:
            revenue = shares * price
            balance += revenue
            trades.append(
                {
                    "Date": row["Date"],
                    "Action": "SELL",
                    "Price": price,
                    "Shares": shares,
                    "Revenue": revenue,
                    "Balance": balance,
                }
            )
            shares = 0

    final_price = data.iloc[-1]["Close"]
    final_value = balance + (shares * final_price)

    return {
        "final_balance": balance,
        "final_shares": shares,
        "final_value": final_value,
        "profit": final_value - initial_balance,
        "roi": ((final_value - initial_balance) / initial_balance) * 100,
        "trades": trades,
    }


# ==================== MAIN APP ====================


def main():
    # Header
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem 0;'>
            <h1>📈 AI Trading Dashboard</h1>
            <p style='color: white; font-size: 1.2rem; opacity: 0.9;'>
                Ultra-Modern Machine Learning Stock Prediction System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem 0;'>
                <h2 style='color: white;'>⚙️ Settings</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Stock selection
        ticker = st.text_input(
            "📊 Stock Ticker",
            value="AAPL",
            help="Enter stock symbol (e.g., AAPL, TSLA, GOOGL)",
        )

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", value=datetime(2015, 1, 1), help="Select start date"
            )
        with col2:
            end_date = st.date_input(
                "End Date", value=datetime.now(), help="Select end date"
            )

        # Load data button
        if st.button("🔄 Load Data", use_container_width=True):
            st.session_state.data_loaded = False

        st.markdown("---")

        # Model selection
        st.markdown(
            "<h3 style='color: white;'>🤖 Model Selection</h3>", unsafe_allow_html=True
        )
        model_choice = st.radio(
            "Choose Model",
            ["Logistic Regression", "Neural Network"],
            help="Select the AI model for predictions",
        )

        st.markdown("---")

        # Trading parameters
        st.markdown(
            "<h3 style='color: white;'>💰 Trading Parameters</h3>",
            unsafe_allow_html=True,
        )
        initial_balance = st.number_input(
            "Initial Balance ($)", value=10000, min_value=1000, step=1000
        )

        st.markdown("---")

        # Info
        st.markdown(
            """
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-top: 2rem;'>
                <p style='color: white; font-size: 0.9rem; margin: 0;'>
                    <strong>📌 Quick Guide:</strong><br>
                    1. Select stock ticker<br>
                    2. Choose date range<br>
                    3. Load data<br>
                    4. Explore predictions<br>
                    5. Analyze performance
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Load data
    if "data" not in st.session_state or not st.session_state.get("data_loaded", False):
        with st.spinner("🔄 Loading stock data..."):
            data = load_stock_data(
                ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            if data is not None:
                st.session_state.data = data
                st.session_state.ticker = ticker
                st.session_state.data_loaded = True
            else:
                st.error(
                    "Failed to load data. Please check the ticker symbol and try again."
                )
                return

    data = st.session_state.data
    ticker = st.session_state.ticker

    # Overview Section
    st.markdown("## 📊 Market Overview")

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    current_price = data["Close"].iloc[-1]
    prev_price = data["Close"].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100

    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
        )

    with col2:
        st.metric("High (52W)", f"${data['High'].max():.2f}")

    with col3:
        st.metric("Low (52W)", f"${data['Low'].min():.2f}")

    with col4:
        avg_volume = data["Volume"].mean()
        st.metric("Avg Volume", f"{avg_volume / 1e6:.2f}M")

    with col5:
        volatility = data["Close"].pct_change().std() * 100
        st.metric("Volatility", f"{volatility:.2f}%")

    # Charts Section
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📈 Price Chart",
            "🔍 Technical Analysis",
            "🤖 AI Predictions",
            "💹 Trading Simulation",
            "📊 Model Performance",
        ]
    )

    with tab1:
        st.markdown("### Stock Price Movement")

        # Candlestick chart
        fig_candle = create_candlestick_chart(
            data.tail(180), f"{ticker} - Last 6 Months"
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        # Volume chart
        fig_volume = create_volume_chart(data.tail(180))
        st.plotly_chart(fig_volume, use_container_width=True)

        # Recent data
        with st.expander("📋 View Recent Data"):
            st.dataframe(
                data.tail(10)[
                    ["Date", "Open", "High", "Low", "Close", "Volume"]
                ].style.format(
                    {
                        "Open": "${:.2f}",
                        "High": "${:.2f}",
                        "Low": "${:.2f}",
                        "Close": "${:.2f}",
                        "Volume": "{:,.0f}",
                    }
                ),
                use_container_width=True,
            )

    with tab2:
        st.markdown("### Technical Indicators Analysis")

        # Technical indicators chart
        fig_tech = create_technical_indicators_chart(data.tail(180))
        st.plotly_chart(fig_tech, use_container_width=True)

        # Indicator summary
        data_indicators = add_technical_indicators(data)
        latest = data_indicators.iloc[-1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📊 RSI Analysis")
            rsi_value = latest["RSI"]
            if rsi_value > 70:
                rsi_status = "🔴 Overbought"
                rsi_color = "red"
            elif rsi_value < 30:
                rsi_status = "🟢 Oversold"
                rsi_color = "green"
            else:
                rsi_status = "🟡 Neutral"
                rsi_color = "orange"

            st.markdown(
                f"""
                <div style='background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px;'>
                    <h2 style='color: {rsi_color}; margin: 0;'>{rsi_value:.2f}</h2>
                    <p style='margin: 0;'>{rsi_status}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("#### 📈 MACD Signal")
            macd = latest["MACD"]
            signal = latest["Signal_Line"]

            if macd > signal:
                macd_status = "🟢 Bullish"
                macd_color = "green"
            else:
                macd_status = "🔴 Bearish"
                macd_color = "red"

            st.markdown(
                f"""
                <div style='background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px;'>
                    <h2 style='color: {macd_color}; margin: 0;'>{macd:.2f}</h2>
                    <p style='margin: 0;'>{macd_status}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown("#### 📉 Volatility")
            vol = latest["Volatility"]
            vol_status = "High" if vol > data_indicators["Volatility"].mean() else "Low"
            vol_color = "red" if vol_status == "High" else "green"

            st.markdown(
                f"""
                <div style='background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px;'>
                    <h2 style='color: {vol_color}; margin: 0;'>{vol:.2f}</h2>
                    <p style='margin: 0;'>{vol_status} Volatility</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab3:
        st.markdown("### AI-Powered Price Prediction")

        # Train models
        with st.spinner("🤖 Training AI models..."):
            models_data = train_models(data)

        # Select model
        if model_choice == "Logistic Regression":
            model = models_data["lr_model"]
            accuracy = models_data["lr_acc"]
            model_type = "Logistic Regression"
        else:
            model = models_data["mlp_model"]
            accuracy = models_data["mlp_acc"]
            model_type = "Neural Network"

        # Make prediction for tomorrow
        st.markdown("#### 🔮 Tomorrow's Prediction")

        if model_choice == "Neural Network":
            # Need to prepare data with technical indicators
            data_pred = add_technical_indicators(data)
            latest_features = data_pred[
                [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "SMA_5",
                    "SMA_20",
                    "RSI",
                    "MACD",
                    "Volatility",
                ]
            ].iloc[-1:]
            latest_scaled = models_data["scaler"].transform(latest_features)
            prediction = model.predict(latest_scaled)[0]
            prediction_proba = model.predict_proba(latest_scaled)[0]
        else:
            latest_features = data[["Open", "High", "Low", "Close", "Volume"]].iloc[-1:]
            prediction = model.predict(latest_features)[0]
            prediction_proba = model.predict_proba(latest_features)[0]

        probability = prediction_proba[1]

        col1, col2 = st.columns([1, 1])

        with col1:
            # Prediction gauge
            fig_gauge = create_prediction_gauge(probability, prediction)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Prediction details
            prediction_text = "📈 UP" if prediction == 1 else "📉 DOWN"
            prediction_color = "green" if prediction == 1 else "red"

            st.markdown(
                f"""
                <div style='background: rgba(255,255,255,0.9); padding: 2rem; border-radius: 15px; text-align: center; margin-top: 2rem;'>
                    <h1 style='color: {prediction_color}; margin: 0; font-size: 3rem;'>{prediction_text}</h1>
                    <p style='font-size: 1.2rem; margin: 1rem 0;'>Next Day Prediction</p>
                    <hr>
                    <p style='font-size: 1rem; color: #666; margin: 0.5rem 0;'>
                        <strong>Model:</strong> {model_type}<br>
                        <strong>Accuracy:</strong> {accuracy * 100:.2f}%<br>
                        <strong>Confidence:</strong> {probability * 100:.2f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Model accuracy
        st.markdown("#### 🎯 Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

            # Confusion matrix
            if model_choice == "Logistic Regression":
                fig_cm = create_confusion_matrix_chart(
                    models_data["y_test_lr"],
                    models_data["lr_pred"],
                    "Logistic Regression - Confusion Matrix",
                )
            else:
                fig_cm = create_confusion_matrix_chart(
                    models_data["y_test_mlp"],
                    models_data["mlp_pred"],
                    "Neural Network - Confusion Matrix",
                )

            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            st.metric("Data Points", f"{len(data):,}")

            # Classification report
            if model_choice == "Logistic Regression":
                report = classification_report(
                    models_data["y_test_lr"], models_data["lr_pred"], output_dict=True
                )
            else:
                report = classification_report(
                    models_data["y_test_mlp"], models_data["mlp_pred"], output_dict=True
                )

            st.markdown("**Classification Report:**")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(
                report_df.style.format("{:.2f}").background_gradient(cmap="RdYlGn"),
                use_container_width=True,
            )

    with tab4:
        st.markdown("### Trading Simulation & Backtesting")

        # Run simulation
        with st.spinner("💹 Running trading simulation..."):
            # Get predictions for simulation
            if model_choice == "Neural Network":
                test_data = add_technical_indicators(data).tail(100)
                X_sim = test_data[
                    [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "SMA_5",
                        "SMA_20",
                        "RSI",
                        "MACD",
                        "Volatility",
                    ]
                ]
                X_sim_scaled = models_data["scaler"].transform(X_sim)
                predictions = model.predict(X_sim_scaled)
            else:
                test_data = data.tail(100)
                X_sim = test_data[["Open", "High", "Low", "Close", "Volume"]]
                predictions = model.predict(X_sim)

            simulation = simulate_trading(test_data, predictions, initial_balance)

        # Simulation results
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Final Value",
                f"${simulation['final_value']:,.2f}",
                f"${simulation['profit']:+,.2f}",
            )

        with col2:
            roi_color = "normal" if simulation["roi"] >= 0 else "inverse"
            st.metric("ROI", f"{simulation['roi']:.2f}%", delta_color=roi_color)

        with col3:
            st.metric("Total Trades", len(simulation["trades"]))

        with col4:
            # Buy and hold comparison
            buy_hold_shares = initial_balance / test_data.iloc[0]["Close"]
            buy_hold_value = buy_hold_shares * test_data.iloc[-1]["Close"]
            buy_hold_profit = buy_hold_value - initial_balance
            st.metric(
                "Buy & Hold", f"${buy_hold_value:,.2f}", f"${buy_hold_profit:+,.2f}"
            )

        # Strategy comparison
        st.markdown("#### 📊 Strategy Comparison")

        comparison_data = pd.DataFrame(
            {
                "Strategy": ["AI Trading", "Buy & Hold"],
                "Final Value": [simulation["final_value"], buy_hold_value],
                "Profit": [simulation["profit"], buy_hold_profit],
                "ROI (%)": [
                    simulation["roi"],
                    (buy_hold_profit / initial_balance) * 100,
                ],
            }
        )

        fig_comparison = px.bar(
            comparison_data,
            x="Strategy",
            y="ROI (%)",
            color="Strategy",
            color_discrete_map={"AI Trading": "#667eea", "Buy & Hold": "#764ba2"},
            title="ROI Comparison",
        )
        fig_comparison.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            height=400,
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Trade history
        with st.expander("📜 View Trade History"):
            if simulation["trades"]:
                trades_df = pd.DataFrame(simulation["trades"])
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No trades executed in this simulation period.")

    with tab5:
        st.markdown("### Model Performance Analysis")

        # Compare all models
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏆 Model Comparison")

            comparison = pd.DataFrame(
                {
                    "Model": ["Logistic Regression", "Neural Network"],
                    "Accuracy": [
                        models_data["lr_acc"] * 100,
                        models_data["mlp_acc"] * 100,
                    ],
                    "Features": [5, 10],
                }
            )

            fig_model_comp = px.bar(
                comparison,
                x="Model",
                y="Accuracy",
                color="Model",
                text="Accuracy",
                color_discrete_map={
                    "Logistic Regression": "#43e97b",
                    "Neural Network": "#fa709a",
                },
            )
            fig_model_comp.update_traces(
                texttemplate="%{text:.2f}%", textposition="outside"
            )
            fig_model_comp.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.1)",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_model_comp, use_container_width=True)

        with col2:
            st.markdown("#### 📈 Performance Metrics")

            metrics_df = pd.DataFrame(
                {
                    "Metric": ["Precision", "Recall", "F1-Score"],
                    "Logistic Regression": [
                        classification_report(
                            models_data["y_test_lr"],
                            models_data["lr_pred"],
                            output_dict=True,
                        )["weighted avg"]["precision"],
                        classification_report(
                            models_data["y_test_lr"],
                            models_data["lr_pred"],
                            output_dict=True,
                        )["weighted avg"]["recall"],
                        classification_report(
                            models_data["y_test_lr"],
                            models_data["lr_pred"],
                            output_dict=True,
                        )["weighted avg"]["f1-score"],
                    ],
                    "Neural Network": [
                        classification_report(
                            models_data["y_test_mlp"],
                            models_data["mlp_pred"],
                            output_dict=True,
                        )["weighted avg"]["precision"],
                        classification_report(
                            models_data["y_test_mlp"],
                            models_data["mlp_pred"],
                            output_dict=True,
                        )["weighted avg"]["recall"],
                        classification_report(
                            models_data["y_test_mlp"],
                            models_data["mlp_pred"],
                            output_dict=True,
                        )["weighted avg"]["f1-score"],
                    ],
                }
            )

            st.dataframe(
                metrics_df.style.format(
                    {"Logistic Regression": "{:.4f}", "Neural Network": "{:.4f}"}
                ).background_gradient(cmap="RdYlGn", axis=1),
                use_container_width=True,
            )

            st.markdown("---")
            st.info(
                """
                **📌 Understanding the Metrics:**
                - **Accuracy**: Overall correct predictions
                - **Precision**: Correct positive predictions
                - **Recall**: Found actual positives
                - **F1-Score**: Balance of precision & recall
                """
            )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 2rem;'>
            <h3 style='color: white;'>⚠️ Disclaimer</h3>
            <p style='color: white; opacity: 0.9;'>
                This dashboard is for educational purposes only. Stock market predictions are inherently uncertain.
                <br>Do not use this for real trading without proper research and risk management.
            </p>
            <p style='color: white; opacity: 0.7; margin-top: 1rem;'>
                Made with ❤️ using Streamlit, Python, and Machine Learning
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
