"""
Market data service — wraps yfinance for all price/fundamental data.
All heavy yfinance calls are run in a thread executor to avoid blocking the event loop.
"""

import asyncio
import logging
from functools import lru_cache
from typing import Optional
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

# Universe of ~50 stocks used by screener and daily report
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "LLY",
    "JPM", "V", "AVGO", "XOM", "PG", "MA", "COST", "HD", "CVX", "MRK",
    "ABBV", "ORCL", "BAC", "KO", "PEP", "AMD", "CSCO", "TMO", "ACN", "MCD",
    "CRM", "NFLX", "TXN", "LIN", "PM", "DHR", "NEE", "ADBE", "WMT", "RTX",
    "T", "INTC", "VZ", "QCOM", "AMGN", "HON", "UPS", "IBM", "CAT", "GE",
]

MACRO_TICKERS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "VIX": "^VIX",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Gold": "GC=F",
    "Oil (WTI)": "CL=F",
}


def _run_sync(func, *args, **kwargs):
    """Run a synchronous function in the default thread pool."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def resolve_ticker(query: str) -> Optional[str]:
    """
    Resolve company name or partial ticker to a valid ticker symbol.
    Returns the ticker string or None if not found.
    """
    query = query.strip()
    # If it looks like a ticker already, validate it
    if len(query) <= 5 and query.isalpha():
        ticker_upper = query.upper()
        try:
            info = await _run_sync(_get_info_sync, ticker_upper)
            if info and info.get("regularMarketPrice") or info.get("currentPrice"):
                return ticker_upper
        except Exception:
            pass

    # Use yfinance search
    try:
        results = await _run_sync(yf.Search, query)
        quotes = getattr(results, "quotes", None) or []
        if quotes:
            return quotes[0].get("symbol", "").upper() or None
    except Exception as e:
        logger.warning(f"resolve_ticker search failed for '{query}': {e}")

    return None


def _get_info_sync(ticker: str) -> dict:
    return yf.Ticker(ticker).info


async def get_stock_info(ticker: str) -> dict:
    """Fetch full yfinance info dict for a ticker."""
    try:
        info = await _run_sync(_get_info_sync, ticker)
        return info or {}
    except Exception as e:
        logger.error(f"get_stock_info({ticker}): {e}")
        return {}


async def get_price(ticker: str) -> Optional[float]:
    """Get current/last price for a ticker."""
    try:
        info = await _run_sync(_get_info_sync, ticker)
        return (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
    except Exception as e:
        logger.error(f"get_price({ticker}): {e}")
        return None


async def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV history. period examples: 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max.
    interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
    """
    def _fetch():
        t = yf.Ticker(ticker)
        return t.history(period=period, interval=interval, auto_adjust=True)

    try:
        df = await _run_sync(_fetch)
        if df is not None and not df.empty:
            return df
        return None
    except Exception as e:
        logger.error(f"get_history({ticker}, {period}): {e}")
        return None


async def get_earnings(ticker: str) -> dict:
    """Returns dict with next_earnings_date and quarterly EPS history."""
    def _fetch():
        t = yf.Ticker(ticker)
        result = {}
        # Upcoming earnings date
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                dates = list(cal.get("Earnings Date", []))
                result["next_earnings_date"] = str(dates[0]) if dates else "N/A"
            else:
                result["next_earnings_date"] = "N/A"
        except Exception:
            result["next_earnings_date"] = "N/A"

        # Quarterly earnings: actual vs estimate
        try:
            eq = t.earnings_dates
            if eq is not None and not eq.empty:
                rows = []
                for date, row in eq.iterrows():
                    eps_est = row.get("EPS Estimate")
                    eps_act = row.get("Reported EPS")
                    surprise = row.get("Surprise(%)")
                    if pd.notna(eps_act):
                        rows.append({
                            "date": str(date.date()),
                            "estimate": round(float(eps_est), 2) if pd.notna(eps_est) else None,
                            "actual": round(float(eps_act), 2),
                            "surprise_pct": round(float(surprise), 1) if pd.notna(surprise) else None,
                        })
                result["quarterly"] = rows[:4]
            else:
                result["quarterly"] = []
        except Exception as e:
            logger.warning(f"earnings quarterly fetch failed for {ticker}: {e}")
            result["quarterly"] = []

        return result

    try:
        return await _run_sync(_fetch)
    except Exception as e:
        logger.error(f"get_earnings({ticker}): {e}")
        return {"next_earnings_date": "N/A", "quarterly": []}


async def get_insider_transactions(ticker: str) -> list[dict]:
    """Fetch recent SEC-filed insider transactions."""
    def _fetch():
        t = yf.Ticker(ticker)
        df = t.insider_transactions
        if df is None or df.empty:
            return []
        rows = []
        for _, row in df.head(15).iterrows():
            rows.append({
                "name": str(row.get("Insider", "Unknown")),
                "title": str(row.get("Position", "")),
                "transaction": str(row.get("Transaction", "")),
                "shares": int(row.get("Shares", 0)) if pd.notna(row.get("Shares")) else 0,
                "value": float(row.get("Value", 0)) if pd.notna(row.get("Value")) else 0,
                "date": str(row.get("Start Date", "")),
                "ownership": str(row.get("Ownership", "")),
            })
        return rows

    try:
        return await _run_sync(_fetch)
    except Exception as e:
        logger.error(f"get_insider_transactions({ticker}): {e}")
        return []


async def get_institutional_holders(ticker: str) -> list[dict]:
    """Fetch top institutional holders."""
    def _fetch():
        t = yf.Ticker(ticker)
        df = t.institutional_holders
        if df is None or df.empty:
            return []
        rows = []
        for _, row in df.head(10).iterrows():
            rows.append({
                "holder": str(row.get("Holder", "")),
                "shares": int(row.get("Shares", 0)) if pd.notna(row.get("Shares")) else 0,
                "value": float(row.get("Value", 0)) if pd.notna(row.get("Value")) else 0,
                "pct_out": float(row.get("% Out", 0)) if pd.notna(row.get("% Out")) else 0,
            })
        return rows

    try:
        return await _run_sync(_fetch)
    except Exception as e:
        logger.error(f"get_institutional_holders({ticker}): {e}")
        return []


async def get_movers() -> dict:
    """Fetch top 5 gainers and losers using Yahoo Finance screener tickers."""
    GAIN_TICKERS = ["NVDA", "AMD", "META", "TSLA", "AMZN", "GOOGL", "NFLX", "AAPL",
                    "MSFT", "AVGO", "CRM", "ADBE", "ORCL", "QCOM", "INTC"]
    LOSE_TICKERS = ["XOM", "CVX", "BA", "GE", "T", "VZ", "IBM", "F", "GM",
                    "WBA", "CVS", "MO", "PM", "KO", "PEP"]

    all_tickers = list(set(GAIN_TICKERS + LOSE_TICKERS + STOCK_UNIVERSE[:20]))

    def _fetch():
        data = yf.download(
            all_tickers,
            period="2d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        changes = []
        for t in all_tickers:
            try:
                if len(all_tickers) == 1:
                    hist = data
                else:
                    hist = data[t] if t in data.columns.get_level_values(0) else None
                if hist is None or hist.empty or len(hist) < 2:
                    continue
                close = hist["Close"].dropna()
                if len(close) < 2:
                    continue
                pct = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
                price = float(close.iloc[-1])
                changes.append({"ticker": t, "price": price, "change_pct": float(pct)})
            except Exception:
                continue
        return sorted(changes, key=lambda x: x["change_pct"], reverse=True)

    try:
        changes = await _run_sync(_fetch)
        gainers = [c for c in changes if c["change_pct"] > 0][:5]
        losers = [c for c in reversed(changes) if c["change_pct"] < 0][:5]
        return {"gainers": gainers, "losers": losers}
    except Exception as e:
        logger.error(f"get_movers: {e}")
        return {"gainers": [], "losers": []}


async def get_macro_data() -> dict:
    """Fetch live macro dashboard data."""
    def _fetch():
        tickers = list(MACRO_TICKERS.values())
        result = {}
        for name, sym in MACRO_TICKERS.items():
            try:
                t = yf.Ticker(sym)
                info = t.info
                price = (
                    info.get("regularMarketPrice")
                    or info.get("currentPrice")
                    or info.get("previousClose", 0)
                )
                prev = info.get("regularMarketPreviousClose") or info.get("previousClose", price)
                change_pct = ((price - prev) / prev * 100) if prev else 0
                result[name] = {
                    "symbol": sym,
                    "price": price,
                    "change_pct": change_pct,
                    "currency": info.get("currency", "USD"),
                }
            except Exception as e:
                logger.warning(f"macro fetch failed for {name} ({sym}): {e}")
                result[name] = {"symbol": sym, "price": None, "change_pct": 0}
        return result

    try:
        return await _run_sync(_fetch)
    except Exception as e:
        logger.error(f"get_macro_data: {e}")
        return {}


async def get_vix() -> Optional[float]:
    """Fetch current VIX level."""
    try:
        info = await _run_sync(_get_info_sync, "^VIX")
        return (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )
    except Exception as e:
        logger.error(f"get_vix: {e}")
        return None


async def get_spy_returns(period: str = "1y") -> Optional[pd.Series]:
    """Fetch SPY daily returns for relative strength calculation."""
    df = await get_history("SPY", period=period)
    if df is not None and not df.empty:
        return df["Close"].pct_change().dropna()
    return None


async def batch_get_prices(tickers: list[str]) -> dict[str, Optional[float]]:
    """Fetch prices for multiple tickers in one batch yfinance download."""
    def _fetch():
        try:
            data = yf.download(
                tickers,
                period="2d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            prices = {}
            for t in tickers:
                try:
                    if len(tickers) == 1:
                        close = data["Close"].dropna()
                    else:
                        close = data[t]["Close"].dropna() if t in data.columns.get_level_values(0) else pd.Series()
                    prices[t] = float(close.iloc[-1]) if not close.empty else None
                except Exception:
                    prices[t] = None
            return prices
        except Exception as e:
            logger.error(f"batch_get_prices download failed: {e}")
            return {t: None for t in tickers}

    return await _run_sync(_fetch)


async def get_analyst_data(ticker: str) -> dict:
    """Fetch analyst ratings counts and price targets from yfinance."""
    def _fetch():
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        result = {
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "target_mean": info.get("targetMeanPrice"),
            "target_median": info.get("targetMedianPrice"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "num_analysts": info.get("numberOfAnalystOpinions"),
            "strong_buy": 0,
            "buy": 0,
            "hold": 0,
            "sell": 0,
            "strong_sell": 0,
        }
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                # Get the most recent period's counts
                latest = recs.iloc[-1] if "strongBuy" in recs.columns else None
                if latest is not None:
                    result["strong_buy"] = int(latest.get("strongBuy", 0))
                    result["buy"] = int(latest.get("buy", 0))
                    result["hold"] = int(latest.get("hold", 0))
                    result["sell"] = int(latest.get("sell", 0))
                    result["strong_sell"] = int(latest.get("strongSell", 0))
        except Exception:
            pass
        return result

    return await _run_sync(_fetch)


async def get_dividend_data(ticker: str) -> dict:
    """Fetch dividend info and recent history from yfinance."""
    def _fetch():
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        result = {
            "dividend_yield": info.get("dividendYield"),
            "dividend_rate": info.get("dividendRate"),
            "ex_dividend_date": None,
            "payout_ratio": info.get("payoutRatio"),
            "five_year_avg_yield": info.get("fiveYearAvgDividendYield"),
            "history": [],
        }
        ex_ts = info.get("exDividendDate")
        if ex_ts:
            import datetime
            try:
                result["ex_dividend_date"] = datetime.datetime.fromtimestamp(ex_ts).strftime("%Y-%m-%d")
            except Exception:
                pass
        try:
            hist = t.dividends
            if hist is not None and not hist.empty:
                recent = hist.tail(8)
                result["history"] = [
                    {"date": str(idx.date()), "amount": round(float(val), 4)}
                    for idx, val in recent.items()
                ]
        except Exception:
            pass
        return result

    return await _run_sync(_fetch)


async def resolve_thai_ticker(query: str) -> Optional[str]:
    """
    Resolve a Thai stock name or ticker to its Yahoo Finance symbol (e.g. PTT.BK).
    Handles: bare ticker (PTT), suffixed (PTT.BK), company name (PTT Public).
    """
    query = query.strip().upper()

    # Already has .BK suffix
    if query.endswith(".BK"):
        try:
            info = await _run_sync(_get_info_sync, query)
            if info and (info.get("regularMarketPrice") or info.get("currentPrice")):
                return query
        except Exception:
            pass
        return None

    # Bare ticker — try appending .BK first (fastest path)
    candidate = f"{query}.BK"
    try:
        info = await _run_sync(_get_info_sync, candidate)
        if info and (info.get("regularMarketPrice") or info.get("currentPrice")):
            return candidate
    except Exception:
        pass

    # Company name — search Yahoo Finance and pick the first .BK result
    try:
        results = await _run_sync(yf.Search, f"{query} Thailand")
        quotes = getattr(results, "quotes", None) or []
        for q in quotes:
            sym = q.get("symbol", "")
            if sym.endswith(".BK"):
                return sym
    except Exception as e:
        logger.warning(f"resolve_thai_ticker search failed for '{query}': {e}")

    # Last resort: plain search, take first .BK hit
    try:
        results = await _run_sync(yf.Search, query)
        quotes = getattr(results, "quotes", None) or []
        for q in quotes:
            sym = q.get("symbol", "")
            if sym.endswith(".BK"):
                return sym
    except Exception:
        pass

    return None


async def get_compare_data(ticker1: str, ticker2: str) -> dict:
    """Fetch side-by-side comparison data for two tickers."""
    async def _get_one(t):
        info = await get_stock_info(t)
        hist = await get_history(t, period="1y")
        return info, hist

    (info1, hist1), (info2, hist2) = await asyncio.gather(
        _get_one(ticker1), _get_one(ticker2)
    )
    return {ticker1: (info1, hist1), ticker2: (info2, hist2)}


async def search_tickers(query: str) -> list[dict]:
    """Search for tickers by name or symbol, returning up to 25 matches for autocomplete."""
    if not query:
        return []
    try:
        results = await _run_sync(yf.Search, query)
        quotes = getattr(results, "quotes", None) or []
        return [
            {
                "symbol": q.get("symbol", ""),
                "name": q.get("shortname") or q.get("longname") or q.get("symbol", ""),
            }
            for q in quotes[:25]
            if q.get("symbol") and q.get("quoteType") in ("EQUITY", "ETF", "MUTUALFUND", None)
        ]
    except Exception as e:
        logger.warning(f"search_tickers('{query}'): {e}")
        return []


async def search_thai_tickers(query: str) -> list[dict]:
    """Search for SET-listed (.BK) tickers for autocomplete."""
    if not query:
        return []
    try:
        q = query.strip().upper()
        results = await _run_sync(yf.Search, f"{q} Thailand")
        quotes = getattr(results, "quotes", None) or []
        thai = [
            {"symbol": r.get("symbol", ""), "name": r.get("shortname") or r.get("longname") or r.get("symbol", "")}
            for r in quotes
            if r.get("symbol", "").endswith(".BK")
        ]
        if not thai:
            results2 = await _run_sync(yf.Search, q)
            quotes2 = getattr(results2, "quotes", None) or []
            thai = [
                {"symbol": r.get("symbol", ""), "name": r.get("shortname") or r.get("longname") or r.get("symbol", "")}
                for r in quotes2
                if r.get("symbol", "").endswith(".BK")
            ]
        return thai[:25]
    except Exception as e:
        logger.warning(f"search_thai_tickers('{query}'): {e}")
        return []
