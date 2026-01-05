#!/bin/bash

echo "=========================================="
echo "🐙 GitHub Repository Setup"
echo "=========================================="
echo ""

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "📝 Initializing git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository already exists"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo ""
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Jupyter
.ipynb_checkpoints

# Environment variables
.env
.env.local

# Large model files (optional - can be regenerated)
*.h5
*.pkl
*.joblib

# Data files (can be regenerated)
stock_data.csv
*.csv

# Logs
*.log
logs/

# OS
Thumbs.db

# Streamlit secrets
.streamlit/secrets.toml

# Backups
*_backup_*/
*.tar.gz

GITIGNORE
    echo "✅ .gitignore created"
fi

# Add all files
echo ""
echo "📦 Adding files to git..."
git add .

# Create initial commit
echo ""
echo "💾 Creating commit..."
COMMIT_MSG="Complete AI Trading Platform - Production Ready

Features:
- Multi-asset support (Stocks, Crypto, Forex)
- 5+ ML models with 56-58% accuracy
- 100+ technical indicators
- Real-time updates and live mode
- Professional UI/UX
- Complete documentation (15+ guides)
- Production ready and deployment ready

Status: ✅ Phase 1 Complete"

git commit -m "$COMMIT_MSG" 2>/dev/null || echo "No changes to commit"

echo ""
echo "=========================================="
echo "✅ Git Repository Ready!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1️⃣  Create GitHub Repository:"
echo "   • Go to: https://github.com/new"
echo "   • Name: ai-trading-platform"
echo "   • Description: AI-powered trading platform with ML predictions"
echo "   • Keep PUBLIC (for free Streamlit deployment)"
echo "   • DON'T initialize with README"
echo "   • Click 'Create repository'"
echo ""
echo "2️⃣  Connect & Push:"
echo "   Run these commands (replace YOUR_USERNAME):"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/ai-trading-platform.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3️⃣  Deploy to Streamlit Cloud:"
echo "   • Go to: https://share.streamlit.io"
echo "   • Sign in with GitHub"
echo "   • Click 'New app'"
echo "   • Select your repository"
echo "   • Main file: phase1_complete.py"
echo "   • Click 'Deploy!'"
echo ""
echo "4️⃣  Your Live URL:"
echo "   https://YOUR_USERNAME-ai-trading-platform.streamlit.app"
echo ""
echo "=========================================="
echo ""
echo "📊 Repository Status:"
git status --short 2>/dev/null || echo "Git not initialized"
echo ""
echo "📝 Commit Log:"
git log --oneline -5 2>/dev/null || echo "No commits yet"
echo ""
echo "=========================================="

