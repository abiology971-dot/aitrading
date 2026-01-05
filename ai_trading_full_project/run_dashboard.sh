#!/bin/bash

# 🚀 AI Trading Dashboard Launcher Script
# This script launches the ultra-modern AI trading dashboard

echo "========================================"
echo "🚀 AI Trading Dashboard Launcher"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found${NC}"

# Check if streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Streamlit not found. Installing dependencies...${NC}"
    pip install -r requirements_dashboard.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Streamlit found${NC}"
fi

# Check if stock_data.csv exists
if [ ! -f "stock_data.csv" ]; then
    echo -e "${YELLOW}⚠️  Stock data not found. Downloading...${NC}"
    python3 fetch_data.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to download stock data${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Stock data downloaded${NC}"
else
    echo -e "${GREEN}✓ Stock data found${NC}"
fi

echo ""
echo "========================================"
echo -e "${BLUE}🎯 Launching Dashboard...${NC}"
echo "========================================"
echo ""
echo -e "${GREEN}Dashboard will open automatically in your browser${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Launch the dashboard
streamlit run dashboard.py

# If streamlit exits
echo ""
echo "========================================"
echo -e "${BLUE}Dashboard stopped${NC}"
echo "========================================"
