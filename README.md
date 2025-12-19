# 🤖 BTC Futures Trading Bot with AI Ensemble

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An advanced AI-powered cryptocurrency trading bot for Binance Futures with a 6-pillar signal ensemble, LLM Overlord decision maker, and real-time dashboard.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Dashboard](#-dashboard) • [Configuration](#-configuration)

![Dashboard Screenshot](screenshots/dashboard.png)

</div>

---

## 🌟 Features

### 🧠 6-Pillar AI Signal Ensemble
The bot combines 6 independent signal sources for robust decision-making:

| Pillar | Technology | Description |
|--------|------------|-------------|
| **ML Prediction** | XGBoost + LightGBM | Ensemble of 2 gradient boosting models predicting price direction |
| **News Sentiment** | Qwen Vision LLM | Analyzes crypto news headlines for market sentiment |
| **Trading Wisdom** | Qwen VL Max (DashScope) | 6 legendary trader personalities provide analysis |
| **Quant Analysis** | Statistical Models | Z-score, momentum, volatility regime, swing detection |
| **HTF Trend** | Multi-Timeframe | 4H + Daily trend alignment filter with override logic |
| **Whale Tracking** | On-Chain Analysis | Monitors large BTC movements and exchange flows |

### 🎯 LLM Overlord (DeepSeek R1)
A final decision-maker powered by DeepSeek R1's reasoning capabilities:
- Analyzes all 6 pillars holistically
- Can override ensemble decisions with high confidence
- Provides detailed reasoning for every decision
- Rate-limited to prevent excessive API calls

### 📊 Real-Time Dashboard
Beautiful Next.js dashboard with:
- Live TradingView chart integration
- Real-time position tracking with P&L
- All signal pillars visualized
- Overlord decision display with confidence ring
- Whale activity alerts
- Emergency close button

### 🛡️ Risk Management
- **10x Leverage** with strict risk controls
- **Dynamic TP/SL**: 1.5-3% take profit, 0.8% stop loss
- **Kelly Criterion**: Position sizing based on historical performance
- **Higher Timeframe Filter**: 4H + Daily trend alignment
- **Swing Detection**: Buy low/sell high at local extremes

### 📈 Self-Learning Engine
- Tracks all trade outcomes with full context
- Calculates win rate, expectancy, Sharpe ratio
- Kelly-based position sizing optimization
- Stores data for ML model retraining

---

## 🏗️ Architecture

```
trading_bot/
├── bot_advanced.py        # Main trading bot orchestrator
├── llm_overlord.py        # DeepSeek R1 final decision maker
├── news_fetcher.py        # News sentiment with Qwen Vision
├── trading_wisdom.py      # 6 legendary trader personalities
├── quant_analysis.py      # Statistical & swing analysis
├── whale_tracker.py       # On-chain whale monitoring
├── learning_engine.py     # Self-learning & trade memory
├── trainer_ensemble.py    # ML model training pipeline
├── dashboard_server.py    # Flask API for dashboard
├── dashboard/             # Next.js frontend
│   ├── pages/index.tsx    # Main dashboard UI
│   └── styles/globals.css # Custom styling
├── models/                # Trained ML models
│   ├── xgb_model.json     # XGBoost model
│   └── lgbm_model.txt     # LightGBM model
├── learning_data/         # Trade history & learning
├── state/                 # Bot state persistence
└── data/                  # Historical price data
```

### Signal Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     MARKET DATA                              │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │    ML    │        │   News   │        │  Wisdom  │
   │ Ensemble │        │Sentiment │        │  Oracle  │
   └────┬─────┘        └────┬─────┘        └────┬─────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │  Quant   │        │  Whale   │        │   HTF    │
   │ Analysis │        │ Tracker  │        │  Trend   │
   └────┬─────┘        └────┬─────┘        └────┬─────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  ENSEMBLE VOTE   │
                    │  (Weighted Sum)  │
                    └────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   LLM OVERLORD   │
                    │  (DeepSeek R1)   │
                    │  Final Decision  │
                    └────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  TRADE EXECUTION │
                    │  (Binance API)   │
                    └──────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ (for dashboard)
- Binance Futures account with API access
- DashScope API key (for Qwen - News & Wisdom)
- (Optional) DeepSeek API key for Overlord

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/btc-trading-bot.git
cd btc-trading-bot/trading_bot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

Additional packages you may need:
```bash
pip install python-binance python-dotenv xgboost lightgbm openai
```

### 4. Install Dashboard Dependencies
```bash
cd dashboard
npm install
cd ..
```

### 5. Configure Environment Variables
Create a `.env` file in the parent directory:
```bash
# Binance API (Required)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# Trading Mode
TRADING_MODE=paper  # 'paper' or 'live'

# LLM APIs
DASHSCOPE_API_KEY=your_dashscope_api_key    # For Qwen (News & Wisdom)
DEEPSEEK_API_KEY=your_deepseek_api_key      # For Overlord (optional)
```

### 6. Train ML Models (Optional)
Pre-trained models are included, but you can retrain:
```bash
python trainer_ensemble.py
```

---

## 🚀 Usage

### Quick Start
The easiest way to run everything:
```bash
./start_dashboard.sh
```

This starts:
1. Next.js dashboard on `http://localhost:3000`
2. Trading bot with API server on `http://localhost:5000`

### Manual Start

**Terminal 1 - Dashboard:**
```bash
cd dashboard
npm run dev
```

**Terminal 2 - Trading Bot:**
```bash
python bot_advanced.py
```

### Paper Trading vs Live Trading
```bash
# In .env file:
TRADING_MODE=paper  # Safe testing mode (no real trades)
TRADING_MODE=live   # Real money (use with caution!)
```

---

## 📺 Dashboard

Access the dashboard at `http://localhost:3000`

### Dashboard Features

| Section | Description |
|---------|-------------|
| **Price Chart** | TradingView widget with BTCUSDC |
| **Overlord Card** | LLM decision with confidence ring |
| **Position Panel** | Current position, P&L, emergency close |
| **Signal Cards** | ML, News, Wisdom, Quant signals |
| **Whale Activity** | Large BTC movements alert |
| **Trade Logs** | Real-time bot activity |

### Overlord Card
The Overlord card shows:
- **Decision**: LONG / SHORT / HOLD
- **Confidence**: 0-100% with visual ring
- **Risk Level**: LOW / MEDIUM / HIGH
- **Reasoning**: Brief explanation
- **Key Factors**: Tags showing decision drivers

---

## ⚙️ Configuration

### Trading Parameters (in `bot_advanced.py`)

```python
# Risk Management
self.stop_loss_pct = 0.008      # 0.8% SL (8% with 10x leverage)
self.take_profit_min = 0.015    # 1.5% minimum TP
self.take_profit_max = 0.030    # 3.0% maximum TP
self.leverage = 10              # Futures leverage

# Position Sizing
self.position_size_pct = 0.20   # 20% of balance per trade
self.min_notional = 120         # Minimum $120 order size

# Signal Thresholds
self.signal_threshold = 0.08    # Minimum vote strength to trade
self.overlord_confidence_threshold = 60  # Min confidence to override
```

### Pillar Weights

```python
# Default weights (sum to 1.0)
weights = {
    'ml': 0.25,      # ML ensemble
    'news': 0.15,    # News sentiment
    'wisdom': 0.25,  # Trading wisdom
    'quant': 0.25,   # Quant analysis
    'htf': 0.10      # Higher timeframe
}
```

### Trading Wisdom Personalities

| Trader | Specialty | Style |
|--------|-----------|-------|
| Jesse Livermore | Momentum & Trends | Aggressive |
| George Soros | Macro & Reflexivity | Contrarian |
| Paul Tudor Jones | Technical Analysis | Balanced |
| Ray Dalio | Risk Parity | Conservative |
| Stanley Druckenmiller | Macro Momentum | Flexible |
| Warren Buffett | Value & Patience | Long-term |

---

## 🔑 API Keys Setup

### Binance Futures
1. Go to Binance → API Management
2. Create new API key
3. Enable **Futures** permission
4. **Disable** withdrawals for safety
5. Add IP whitelist

### DeepSeek (for Overlord)
1. Sign up at [DeepSeek](https://platform.deepseek.com/)
2. Get API key from dashboard
3. Add to `.env` as `DEEPSEEK_API_KEY`

### DashScope (for Qwen - News & Wisdom)
1. Sign up at [Alibaba Cloud DashScope](https://dashscope.aliyun.com/)
2. Get API key from console
3. Add to `.env` as `DASHSCOPE_API_KEY`
4. Used for both News Sentiment and Trading Wisdom pillars

---

## 📈 Codebase Stats

| Language | Files | Lines of Code |
|----------|-------|---------------|
| Python | 11 | 5,455 |
| TypeScript/TSX | 3 | 1,649 |
| JSON | 13 | 677 |
| CSS | 1 | 323 |
| Markdown | 1 | 307 |
| JavaScript | 3 | 49 |
| Shell | 1 | 22 |
| **Total** | **33** | **8,482** |



---

## 📊 Performance Metrics

The learning engine tracks:
- **Win Rate**: Percentage of profitable trades
- **Expectancy**: Average profit per trade
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Gross profit / Gross loss
- **Kelly Fraction**: Optimal position size

View in dashboard header or `learning_data/trade_history.json`.

---

## ⚠️ Risk Warnings

> **IMPORTANT**: This bot trades with real money on Binance Futures with leverage. You can lose your entire investment.

1. **Start with paper trading** - Test thoroughly before going live
2. **Never risk more than you can afford to lose**
3. **10x leverage amplifies both gains AND losses**
4. **Monitor the bot** - Don't leave it completely unattended
5. **Secure your API keys** - Never commit them to git
6. **Use IP whitelist** on Binance API
7. **Disable withdrawals** on your API key

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Insufficient balance` | Ensure $12+ USDC in Futures wallet |
| `Order's notional must be no smaller than 100` | Bot auto-adjusts, check balance |
| `API Connection Error` | Check API keys and IP whitelist |
| `Overlord disabled` | Set `DEEPSEEK_API_KEY` in `.env` |
| `Model not found` | Run `python trainer_ensemble.py` |
| `Dashboard not loading` | Run `npm install` in dashboard folder |

### Logs
- Bot logs print to terminal
- Dashboard logs visible in browser console
- Trade history in `learning_data/trade_history.json`

---

## � Future Improvements

### Short-term
- [ ] **Multi-Asset Support** - Extend beyond BTC to ETH, SOL, and other major pairs
- [ ] **Backtesting Dashboard** - Visual backtesting interface with strategy comparison
- [ ] **Telegram/Discord Alerts** - Real-time trade notifications and commands
- [ ] **Portfolio Tracking** - Track overall performance across multiple positions

### Medium-term
- [ ] **Reinforcement Learning Agent** - RL-based position sizing and entry optimization
- [ ] **Order Flow Analysis** - CVD, delta, and footprint chart integration
- [ ] **Sentiment Aggregator** - Twitter/X, Reddit, and Fear & Greed index integration
- [ ] **Auto-Retraining Pipeline** - Scheduled model retraining with fresh market data

### Long-term
- [ ] **Multi-Exchange Support** - Bybit, OKX, Kraken integration
- [ ] **Options Strategies** - Hedging with BTC options for risk management
- [ ] **Distributed Architecture** - Kubernetes deployment for high availability
- [ ] **Mobile App** - React Native companion app for monitoring

### Research Ideas
- [ ] **Transformer Price Prediction** - Replace LSTM with modern transformer architecture
- [ ] **Graph Neural Networks** - Model crypto market correlations
- [ ] **Causal Inference** - Better understand what drives price movements
- [ ] **Adversarial Testing** - Stress test strategies against market manipulation scenarios

---

## �📄 License

MIT License - Use at your own risk.

---

## 🙏 Acknowledgments

- **Ivan Dimitrov** ([@Demimagic-ML](https://github.com/Demimagic-ML)) - Creator & Developer
- [Binance API](https://binance-docs.github.io/apidocs/)
- [DeepSeek](https://www.deepseek.com/) for R1 reasoning model
- [TradingView](https://www.tradingview.com/) for charts
- Legendary traders whose wisdom inspired this project

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the crypto trading community

</div>
