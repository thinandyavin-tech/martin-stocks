# Martin's Stocks — Discord AI Stock Analysis Bot

[![tests](https://github.com/thinandyavin-tech/martin-stocks/actions/workflows/tests.yml/badge.svg)](https://github.com/thinandyavin-tech/martin-stocks/actions/workflows/tests.yml)


A fully autonomous Discord bot for stock analysis powered by Groq LLaMA 3.3 70B, yfinance, and real-time market data.

## Features

- **Daily AI Report** — Morning briefing with AI-scored signals for 50 stocks
- **`/stock`** — Deep dive: technical analysis, smart money, news, AI thesis, interactive chart with 7 timeframe buttons
- **`/movers`** — Top 5 gainers and losers
- **`/macro`** — Live dashboard: indices, VIX, crypto, commodities
- **`/fear`** — VIX-based fear/greed meter with ASCII bar
- **`/screener`** — Scan 50 stocks for Oversold Bounce, Momentum, Breakout, Overbought
- **`/compare`** — Side-by-side comparison of two stocks
- **`/earnings`** — Next earnings date + last 4 quarters EPS vs estimates
- **`/insider`** — Recent SEC-filed insider transactions
- **`/watchlist`** — Personal watchlist with live prices (max 20 tickers)
- **`/alert`** — Price alerts via APScheduler, pings you in channel
- **`/news`** — Top headlines from Reuters, CNBC, MarketWatch with AI summarize buttons
- **`/summarize`** — Paste any URL for an AI summary
- **`/twitter`** — Monitor @ShayBoloor via free Nitter RSS (no API key required)

## Setup

### 1. Prerequisites

- Python 3.11+
- A Discord bot token ([Discord Developer Portal](https://discord.com/developers/applications))
- A Groq API key ([console.groq.com](https://console.groq.com))

### 2. Clone & Install

```bash
git clone https://github.com/thinandyavin-tech/martin-stocks.git
cd martin-stocks
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in DISCORD_TOKEN and GROQ_API_KEY
```

### 4. Discord Bot Permissions

In the Discord Developer Portal, enable these **Bot Permissions**:
- Send Messages
- Embed Links
- Attach Files
- Read Message History
- Use Slash Commands
- Mention Everyone (for alert pings)

Enable **Privileged Gateway Intents**:
- Message Content Intent

### 5. Run

```bash
python main.py
```

On first run, slash commands are synced globally (may take up to 1 hour to propagate; use guild-specific sync for instant dev updates).

## Configuration

### Daily Report

Use `/report set` (in any channel) to designate that channel to receive the daily AI report at 8:00 AM UTC.

### Twitter Monitoring

Use `/twitter on` in any channel to start receiving @ShayBoloor tweets. Use `/twitter off` to stop.

## Project Structure

```
martin_stocks/
├── main.py                  # Bot entry point
├── .env                     # Secrets (not committed)
├── requirements.txt
├── data/
│   └── martin_stocks.db     # SQLite database (auto-created)
├── db/
│   └── database.py          # Async SQLite CRUD layer
├── services/
│   ├── market_data.py       # yfinance wrapper
│   ├── groq_service.py      # Groq LLaMA AI service
│   ├── news_service.py      # RSS feed scraper
│   └── twitter_service.py   # Nitter RSS Twitter monitor
├── utils/
│   ├── technical.py         # EMA, RSI, OBV, candlestick patterns
│   ├── charts.py            # Matplotlib chart generation
│   └── helpers.py           # Formatting utilities
├── cogs/
│   ├── stock_commands.py    # /stock, /compare, /earnings, /insider
│   ├── market_commands.py   # /movers, /macro, /fear, /screener
│   ├── personal_tools.py    # /watchlist, /alert, /news, /summarize
│   ├── twitter_cog.py       # /twitter on|off|status
│   └── daily_report.py      # Daily AI report + /report set
└── scheduler/
    └── tasks.py             # APScheduler: price alerts, Twitter poll, daily report
```

## Commands Reference

| Command | Description |
|---|---|
| `/stock [query]` | Full analysis for any ticker or company name |
| `/compare [stock1] [stock2]` | Side-by-side comparison |
| `/earnings [ticker]` | Earnings calendar + EPS history |
| `/insider [ticker]` | Recent insider transactions |
| `/movers` | Top 5 gainers and losers |
| `/macro` | Live macro dashboard |
| `/fear` | Fear & Greed meter |
| `/screener` | Scan 50 stocks for setups |
| `/watchlist add/remove/show` | Manage your watchlist |
| `/alert set/list/remove` | Manage price alerts |
| `/news` | Top headlines with AI summaries |
| `/summarize [url]` | AI summary of any article |
| `/twitter on/off/status` | Twitter monitoring toggle |
| `/report set` | Set this channel for daily AI reports |

## Notes

- Nitter RSS uses 6 fallback instances. If all fail, Twitter monitoring degrades gracefully.
- yfinance data may have a 15-minute delay for real-time prices.
- Groq LLaMA 3.3 70B is used for all AI analysis — fast and free on the generous Groq free tier.
- SQLite database is stored in `data/martin_stocks.db` (auto-created on first run).
