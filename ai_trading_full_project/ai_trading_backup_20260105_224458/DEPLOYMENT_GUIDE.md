# 🚀 Deployment Guide - AI Trading Dashboard

Complete guide to deploy your ultra-modern AI Trading Dashboard to the cloud!

---

## 📋 Table of Contents
1. [Quick Local Setup](#quick-local-setup)
2. [Streamlit Cloud Deployment (FREE)](#streamlit-cloud-deployment)
3. [Heroku Deployment](#heroku-deployment)
4. [Docker Deployment](#docker-deployment)
5. [AWS Deployment](#aws-deployment)
6. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Local Setup

### Step 1: Install Dependencies
```bash
cd ai_trading_full_project
pip install -r requirements_dashboard.txt
```

### Step 2: Run the Dashboard
```bash
streamlit run dashboard.py
```

The dashboard will open automatically at `http://localhost:8501`

**That's it! Your dashboard is running locally!** 🎉

---

## 🌐 Streamlit Cloud Deployment (FREE & RECOMMENDED)

### Why Streamlit Cloud?
- ✅ **100% FREE** (no credit card required)
- ✅ Super easy deployment (3 minutes)
- ✅ Automatic updates from GitHub
- ✅ Built-in SSL (HTTPS)
- ✅ No configuration needed

### Prerequisites
- GitHub account
- Git installed on your computer

### Step-by-Step Deployment

#### 1. Create GitHub Repository
```bash
cd ai_trading_full_project

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: AI Trading Dashboard"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/ai-trading-dashboard.git
git branch -M main
git push -u origin main
```

#### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in the details:
   - **Repository**: `YOUR_USERNAME/ai-trading-dashboard`
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
   - **Python version**: `3.11`
5. Click **"Deploy!"**

#### 3. Configure (Optional)
In the Streamlit Cloud dashboard:
- Go to **"Advanced settings"**
- Add any secrets if needed
- Configure custom subdomain

#### 4. Access Your Dashboard
Your app will be live at:
```
https://YOUR_USERNAME-ai-trading-dashboard.streamlit.app
```

**🎉 DONE! Your dashboard is now live and accessible worldwide!**

---

## 🔧 Heroku Deployment

### Prerequisites
- Heroku account (free tier available)
- Heroku CLI installed

### Step 1: Prepare Files

Ensure you have these files:
- `dashboard.py` ✅
- `requirements_dashboard.txt` ✅
- `Procfile` ✅
- `.streamlit/config.toml` ✅

### Step 2: Create Heroku App
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-trading-dashboard

# Add buildpack
heroku buildpacks:set heroku/python
```

### Step 3: Deploy
```bash
# Initialize git (if not done)
git init
git add .
git commit -m "Deploy to Heroku"

# Push to Heroku
git push heroku main
```

### Step 4: Open Your App
```bash
heroku open
```

Your dashboard will be at: `https://your-trading-dashboard.herokuapp.com`

### Heroku Configuration
```bash
# Scale dynos
heroku ps:scale web=1

# View logs
heroku logs --tail

# Set environment variables (if needed)
heroku config:set VARIABLE_NAME=value
```

---

## 🐳 Docker Deployment

### Step 1: Create Dockerfile
Create a file named `Dockerfile` in your project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_dashboard.txt .
RUN pip install --no-cache-dir -r requirements_dashboard.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build Docker Image
```bash
docker build -t ai-trading-dashboard .
```

### Step 3: Run Container
```bash
docker run -p 8501:8501 ai-trading-dashboard
```

Access at: `http://localhost:8501`

### Step 4: Push to Docker Hub (Optional)
```bash
# Tag image
docker tag ai-trading-dashboard YOUR_USERNAME/ai-trading-dashboard:latest

# Push to Docker Hub
docker push YOUR_USERNAME/ai-trading-dashboard:latest
```

### Docker Compose (Optional)
Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    volumes:
      - ./:/app
```

Run with:
```bash
docker-compose up -d
```

---

## ☁️ AWS Deployment

### Option 1: AWS EC2

#### Step 1: Launch EC2 Instance
1. Go to AWS Console → EC2
2. Launch new instance (Ubuntu 22.04 LTS)
3. Instance type: t2.micro (free tier)
4. Security Group: Allow port 8501

#### Step 2: Connect and Setup
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3-pip python3-venv -y

# Clone your repository
git clone https://github.com/YOUR_USERNAME/ai-trading-dashboard.git
cd ai-trading-dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_dashboard.txt

# Run with nohup (keeps running after logout)
nohup streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0 &
```

#### Step 3: Access Dashboard
```
http://YOUR_EC2_PUBLIC_IP:8501
```

### Option 2: AWS Elastic Beanstalk

#### Step 1: Install EB CLI
```bash
pip install awsebcli
```

#### Step 2: Initialize
```bash
eb init -p python-3.11 ai-trading-dashboard --region us-east-1
```

#### Step 3: Create Environment
```bash
eb create ai-trading-env
```

#### Step 4: Deploy
```bash
eb deploy
```

#### Step 5: Open App
```bash
eb open
```

---

## 🌍 Other Deployment Options

### Google Cloud Platform (GCP)

#### Cloud Run (Recommended)
```bash
# Build and deploy
gcloud run deploy ai-trading-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Microsoft Azure

#### Azure App Service
```bash
# Login
az login

# Create resource group
az group create --name ai-trading-rg --location eastus

# Create app service plan
az appservice plan create --name ai-trading-plan --resource-group ai-trading-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group ai-trading-rg --plan ai-trading-plan --name ai-trading-dashboard --runtime "PYTHON:3.11"

# Deploy
az webapp up --runtime PYTHON:3.11 --sku B1
```

### DigitalOcean App Platform

1. Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
2. Click **"Create App"**
3. Connect your GitHub repository
4. Configure:
   - **Run Command**: `streamlit run dashboard.py`
   - **HTTP Port**: `8501`
5. Click **"Launch App"**

---

## 🔒 Environment Variables & Secrets

### Local Development
Create `.env` file:
```bash
# API Keys (if needed)
ALPHA_VANTAGE_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# Database (if needed)
DATABASE_URL=your_database_url
```

### Streamlit Cloud Secrets
1. Go to your app settings
2. Click **"Secrets"**
3. Add in TOML format:
```toml
[api_keys]
alpha_vantage = "your_key"
finnhub = "your_key"
```

Access in code:
```python
import streamlit as st
api_key = st.secrets["api_keys"]["alpha_vantage"]
```

### Heroku Config Vars
```bash
heroku config:set API_KEY=your_key_here
```

---

## 🔧 Performance Optimization

### 1. Caching
Already implemented in `dashboard.py`:
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data(ticker, start_date, end_date):
    # Your code here
```

### 2. Data Loading
- Use session state to avoid reloading
- Implement lazy loading for heavy computations
- Compress large datasets

### 3. Image Optimization
```python
# Use WebP format for images
# Compress before uploading
```

### 4. CDN (for production)
- Use CloudFlare for static assets
- Enable browser caching

---

## 📊 Monitoring & Analytics

### Streamlit Built-in Analytics
- View usage in Streamlit Cloud dashboard
- Track unique visitors and pageviews

### Google Analytics Integration
Add to `dashboard.py`:
```python
# Add Google Analytics tracking code
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_TRACKING_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_TRACKING_ID');
</script>
""", unsafe_allow_html=True)
```

### Error Tracking
```bash
# Add Sentry for error tracking
pip install sentry-sdk
```

```python
import sentry_sdk
sentry_sdk.init(dsn="YOUR_SENTRY_DSN")
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:**
```bash
pip install -r requirements_dashboard.txt
```

### Issue: "Port already in use"
**Solution:**
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run dashboard.py --server.port=8502
```

### Issue: "Memory Error"
**Solution:**
- Increase server memory
- Optimize data loading (use sampling)
- Clear cache more frequently

### Issue: "Slow Loading"
**Solution:**
- Reduce data size (use date ranges)
- Implement pagination
- Use caching effectively
- Optimize queries

### Issue: "Deployment Failed"
**Solution:**
```bash
# Check logs
heroku logs --tail  # For Heroku
streamlit cloud logs  # For Streamlit Cloud

# Verify requirements.txt
pip freeze > requirements_dashboard.txt
```

### Issue: "CORS Error"
**Solution:**
Update `.streamlit/config.toml`:
```toml
[server]
enableCORS = true
enableXsrfProtection = false
```

---

## 🔐 Security Best Practices

### 1. API Keys
- ✅ Use environment variables
- ✅ Never commit keys to GitHub
- ✅ Use `.gitignore` for sensitive files

### 2. Input Validation
```python
# Validate ticker input
if not ticker.isalpha() or len(ticker) > 5:
    st.error("Invalid ticker symbol")
```

### 3. Rate Limiting
```python
# Implement rate limiting for API calls
import time
time.sleep(1)  # Wait 1 second between calls
```

### 4. HTTPS
- Always use HTTPS in production
- Streamlit Cloud provides this automatically

---

## 📝 Post-Deployment Checklist

- [ ] Test all features in production
- [ ] Verify data is loading correctly
- [ ] Check mobile responsiveness
- [ ] Test on different browsers
- [ ] Set up monitoring/alerts
- [ ] Configure backup strategy
- [ ] Update documentation
- [ ] Share with users!

---

## 🎯 Quick Command Reference

```bash
# Local development
streamlit run dashboard.py

# With custom port
streamlit run dashboard.py --server.port=8502

# With custom config
streamlit run dashboard.py --server.headless true

# Clear cache
streamlit cache clear

# Check Streamlit version
streamlit --version

# View Streamlit config
streamlit config show
```

---

## 🌟 Production Deployment Checklist

### Before Deployment
- [ ] Test locally thoroughly
- [ ] Update requirements.txt
- [ ] Set up .gitignore
- [ ] Remove debug code
- [ ] Optimize performance
- [ ] Add error handling
- [ ] Write documentation

### During Deployment
- [ ] Choose deployment platform
- [ ] Configure environment
- [ ] Set up secrets/variables
- [ ] Deploy application
- [ ] Test deployment

### After Deployment
- [ ] Verify all features work
- [ ] Set up monitoring
- [ ] Configure analytics
- [ ] Share URL with users
- [ ] Plan maintenance schedule

---

## 💡 Tips for Success

1. **Start Simple**: Deploy to Streamlit Cloud first (easiest)
2. **Test Locally**: Always test before deploying
3. **Use Caching**: Cache data to improve performance
4. **Monitor Usage**: Track users and errors
5. **Update Regularly**: Keep dependencies updated
6. **Backup Data**: Regular backups of important data
7. **Document**: Keep this guide updated

---

## 📞 Support Resources

### Streamlit
- [Streamlit Docs](https://docs.streamlit.io)
- [Community Forum](https://discuss.streamlit.io)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

### Heroku
- [Heroku Dev Center](https://devcenter.heroku.com)
- [Heroku Status](https://status.heroku.com)

### AWS
- [AWS Documentation](https://docs.aws.amazon.com)
- [AWS Forums](https://forums.aws.amazon.com)

---

## 🎉 You're Ready to Deploy!

Choose your preferred method and follow the steps. 

**Recommended for beginners**: Streamlit Cloud (FREE and easiest)

**For production**: AWS, GCP, or Azure

**For containerized apps**: Docker + Kubernetes

---

**Made with ❤️ | Happy Deploying! 🚀**