#!/bin/bash

echo "=========================================="
echo "🎯 Advanced AI Trading System"
echo "=========================================="
echo ""
echo "📚 Installing required packages..."
pip install -q xgboost lightgbm ta-lib-binary 2>&1 | grep -E "Successfully|ERROR" || echo "Packages ready"
echo ""
echo "🚀 Features:"
echo "   • 500+ Advanced Features"
echo "   • 10+ ML Models (XGBoost, LightGBM, etc.)"
echo "   • Time Series Cross-Validation"
echo "   • Ensemble Meta-Learning"
echo "   • Target: 60-75% Accuracy"
echo ""
echo "📊 This will take 5-10 minutes to train..."
echo ""
python advanced_ai_system.py
