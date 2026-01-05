#!/bin/bash

# 🚀 AI Trading Dashboard - Web Deployment Script
# This script automates deployment to Streamlit Cloud

echo "=========================================="
echo "🚀 AI Trading Dashboard - Web Deployment"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed!${NC}"
    echo "Please install Git: https://git-scm.com/downloads"
    exit 1
fi

echo -e "${GREEN}✓ Git found${NC}"

# Check if repository is already initialized
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}⚠️  Git repository not initialized${NC}"
    echo -e "${CYAN}Initializing git repository...${NC}"
    git init
    echo -e "${GREEN}✓ Git initialized${NC}"
else
    echo -e "${GREEN}✓ Git repository exists${NC}"
fi

# Add all files
echo -e "${CYAN}Adding files to git...${NC}"
git add .

# Commit
echo -e "${CYAN}Creating commit...${NC}"
git commit -m "Deploy AI Trading Dashboard to web" || echo -e "${YELLOW}⚠️  No changes to commit${NC}"

echo ""
echo "=========================================="
echo -e "${BLUE}📋 Next Steps for Deployment:${NC}"
echo "=========================================="
echo ""

echo -e "${CYAN}OPTION 1: Streamlit Cloud (FREE & EASIEST)${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "1️⃣  Create a GitHub repository:"
echo "   • Go to https://github.com/new"
echo "   • Name: ai-trading-dashboard"
echo "   • Keep it public"
echo "   • DO NOT initialize with README"
echo "   • Click 'Create repository'"
echo ""
echo "2️⃣  Push your code to GitHub:"
echo -e "   ${YELLOW}git remote add origin https://github.com/YOUR_USERNAME/ai-trading-dashboard.git${NC}"
echo -e "   ${YELLOW}git branch -M main${NC}"
echo -e "   ${YELLOW}git push -u origin main${NC}"
echo ""
echo "3️⃣  Deploy to Streamlit Cloud:"
echo "   • Go to https://share.streamlit.io"
echo "   • Click 'Sign in with GitHub'"
echo "   • Click 'New app'"
echo "   • Repository: YOUR_USERNAME/ai-trading-dashboard"
echo "   • Branch: main"
echo "   • Main file path: dashboard.py"
echo "   • Click 'Deploy!'"
echo ""
echo "4️⃣  Your dashboard will be live at:"
echo -e "   ${GREEN}https://YOUR_USERNAME-ai-trading-dashboard.streamlit.app${NC}"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${CYAN}OPTION 2: Heroku (More Control)${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "1️⃣  Install Heroku CLI:"
echo "   • Download from https://devcenter.heroku.com/articles/heroku-cli"
echo ""
echo "2️⃣  Login and create app:"
echo -e "   ${YELLOW}heroku login${NC}"
echo -e "   ${YELLOW}heroku create your-trading-dashboard${NC}"
echo ""
echo "3️⃣  Deploy:"
echo -e "   ${YELLOW}git push heroku main${NC}"
echo -e "   ${YELLOW}heroku open${NC}"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${CYAN}OPTION 3: Manual GitHub Setup${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "If you want to set up GitHub manually:"
echo ""
echo "1️⃣  Create GitHub repository at: https://github.com/new"
echo ""
echo "2️⃣  Run these commands (replace YOUR_USERNAME):"
echo ""
echo -e "${YELLOW}git remote add origin https://github.com/YOUR_USERNAME/ai-trading-dashboard.git${NC}"
echo -e "${YELLOW}git branch -M main${NC}"
echo -e "${YELLOW}git push -u origin main${NC}"
echo ""
echo "3️⃣  Then deploy via Streamlit Cloud (see Option 1, step 3)"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "=========================================="
echo -e "${BLUE}📚 Documentation Available:${NC}"
echo "=========================================="
echo ""
echo "• DEPLOYMENT_GUIDE.md - Complete deployment guide"
echo "• LAUNCH_GUIDE.md - Quick start guide"
echo "• DASHBOARD_README.md - Dashboard features"
echo ""

echo "=========================================="
echo -e "${GREEN}✅ Repository prepared for deployment!${NC}"
echo "=========================================="
echo ""
echo -e "${CYAN}💡 TIP: Streamlit Cloud is 100% FREE and takes just 3 minutes!${NC}"
echo ""
echo -e "${YELLOW}Need help? Check DEPLOYMENT_GUIDE.md${NC}"
echo ""
