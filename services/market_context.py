"""
Market intelligence context service.

Continuously tracks macro indicators, market regime, sector rotation, and
key economic signals. Results are cached for 15 minutes and injected into
every AI analysis prompt so the model has full awareness of current conditions.
"""
from __future__ import annotations

import asyncio
import logging
import time

import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_TTL = 900  # 15 minutes
_cache: dict = {}
_cache_ts: float = 0.0
_lock = asyncio.Lock()

# 11 SPDR sector ETFs
SECTOR_ETFS: dict[str, str] = {
    "Technology":       "XLK",
    "Healthcare":       "XLV",
    "Financials":       "XLF",
    "Energy":           "XLE",
    "Consumer Discr.":  "XLY",
    "Consumer Staples": "XLP",
    "Industrials":      "XLI",
    "Communication":    "XLC",
    "Materials":        "XLB",
    "Utilities":        "XLU",
    "Real Estate":      "XLRE",
}

# Macro symbols we track
_MACRO: dict[str, str] = {
    "SPY":       "S&P 500",
    "QQQ":       "NASDAQ 100",
    "^VIX":      "VIX",
    "^TNX":      "10Y Yield",
    "DX-Y.NYB":  "DXY Dollar",
    "GC=F":      "Gold",
    "CL=F":      "Oil (WTI)",
    "BTC-USD":   "Bitcoin",
}

# Maps each stock in STOCK_UNIVERSE to its sector ETF for context injection
TICKER_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AMD": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "QCOM": "Technology", "INTC": "Technology",
    "UNH": "Healthcare", "LLY": "Healthcare", "JNJ": "Healthcare",
    "MRK": "Healthcare", "ABBV": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "JPM": "Financials", "BAC": "Financials", "V": "Financials",
    "MA": "Financials", "GS": "Financials", "MS": "Financials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "VLO": "Energy",
    "TSLA": "Consumer Discr.", "AMZN": "Consumer Discr.", "HD": "Consumer Discr.",
    "MCD": "Consumer Discr.", "NKE": "Consumer Discr.",
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "CAT": "Industrials", "HON": "Industrials", "GE": "Industrials",
    "RTX": "Industrials", "UPS": "Industrials", "DE": "Industrials", "LMT": "Industrials",
    "NFLX": "Communication", "DIS": "Communication",
    "T": "Communication", "VZ": "Communication",
}


def _run_sync(func, *args, **kwargs):
    """Run a blocking function in the default thread pool."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def get_market_context() -> dict:
    """Return cached market context, refreshing if older than 15 minutes."""
    global _cache, _cache_ts
    async with _lock:
        now = time.monotonic()
        if _cache and (now - _cache_ts) < _CACHE_TTL:
            return _cache
        try:
            ctx = await _build_context()
            _cache = ctx
            _cache_ts = now
            logger.info(
                f"Market context refreshed — regime={ctx.get('regime')}  "
                f"VIX={ctx.get('vix')}  SPY={ctx.get('spy_change_pct', 0):+.2f}%"
            )
            return ctx
        except Exception as e:
            logger.error(f"market_context build failed: {e}")
            return _cache or {}


async def warm_cache() -> None:
    """Pre-warm the cache on startup so the first analysis is fast."""
    try:
        await get_market_context()
    except Exception as e:
        logger.warning(f"market_context warm_cache: {e}")


async def _build_context() -> dict:
    """Fetch all market intelligence concurrently."""
    macro_raw, sectors_raw = await asyncio.gather(
        _run_sync(_fetch_macro),
        _run_sync(_fetch_sectors),
    )

    spy = macro_raw.get("SPY", {})
    vix_price = macro_raw.get("^VIX", {}).get("price") or 20.0
    vix_chg = macro_raw.get("^VIX", {}).get("change_pct", 0)
    tny = macro_raw.get("^TNX", {}).get("price")
    dxy = macro_raw.get("DX-Y.NYB", {}).get("price")
    dxy_chg = macro_raw.get("DX-Y.NYB", {}).get("change_pct", 0)
    spy_price = spy.get("price", 0)
    spy_ma200 = spy.get("ma200")
    spy_chg = spy.get("change_pct", 0)

    above_200ma = bool(spy_price and spy_ma200 and spy_price > spy_ma200)
    below_pct = round(((spy_price - spy_ma200) / spy_ma200 * 100), 1) if (spy_price and spy_ma200) else 0

    # Regime classification: VIX dominates when extreme
    if vix_price >= 35:
        regime = "PANIC"
        regime_label = (
            f"Market Panic (VIX {vix_price:.1f}) — extreme fear, high risk of further selling. "
            f"Only oversold bounces viable; avoid new longs."
        )
    elif vix_price >= 25:
        regime = "HIGH_VOLATILITY"
        regime_label = (
            f"High Volatility (VIX {vix_price:.1f}) — elevated uncertainty. "
            f"Be selective; favour defensive sectors and tight stops."
        )
    elif vix_price >= 18:
        if above_200ma:
            regime = "CAUTIOUS_BULL"
            regime_label = (
                f"Cautious Bull (VIX {vix_price:.1f}, SPY {below_pct:+.1f}% vs 200MA) — "
                f"uptrend intact but volatility rising. Monitor closely."
            )
        else:
            regime = "ELEVATED_VOLATILITY"
            regime_label = (
                f"Elevated Volatility/Correction (VIX {vix_price:.1f}, SPY {below_pct:+.1f}% vs 200MA) — "
                f"caution warranted; breadth deteriorating."
            )
    elif above_200ma:
        regime = "BULL"
        regime_label = (
            f"Bull Market (VIX {vix_price:.1f}, SPY {below_pct:+.1f}% vs 200MA) — "
            f"risk-on environment; momentum and growth strategies favoured."
        )
    else:
        regime = "BEAR"
        regime_label = (
            f"Bear/Correction (VIX {vix_price:.1f}, SPY {below_pct:+.1f}% vs 200MA) — "
            f"defensive positioning; avoid catching falling knives."
        )

    # Rate environment
    if tny:
        if tny > 5.0:
            rate_env = f"HIGH ({tny:.2f}%) — rate-sensitive stocks under pressure; favour value over growth"
        elif tny > 4.0:
            rate_env = f"ELEVATED ({tny:.2f}%) — headwind for high-multiple tech; financials may benefit"
        else:
            rate_env = f"MODERATE ({tny:.2f}%) — broadly supportive for equities"
    else:
        rate_env = "N/A"

    # Dollar context
    if dxy:
        dollar_note = (
            "STRONG dollar — headwind for multinationals and commodities"
            if dxy_chg > 0.3
            else "WEAK dollar — tailwind for exports, commodities, and EM" if dxy_chg < -0.3
            else "STABLE dollar — neutral impact"
        )
    else:
        dollar_note = "N/A"

    # Sort sectors by today's performance
    sorted_sectors = sorted(
        [(k, v) for k, v in sectors_raw.items() if v.get("change_pct") is not None],
        key=lambda x: x[1]["change_pct"],
        reverse=True,
    )
    hot_sectors  = [(k, v["change_pct"]) for k, v in sorted_sectors[:4]]
    cold_sectors = [(k, v["change_pct"]) for k, v in sorted_sectors[-4:]]

    return {
        "regime":           regime,
        "regime_label":     regime_label,
        "above_200ma":      above_200ma,
        "spy_price":        spy_price,
        "spy_ma200":        spy_ma200,
        "spy_change_pct":   spy_chg,
        "vix":              vix_price,
        "vix_change":       vix_chg,
        "ten_year_yield":   tny,
        "rate_environment": rate_env,
        "dxy":              dxy,
        "dxy_change":       dxy_chg,
        "dollar_note":      dollar_note,
        "gold":             macro_raw.get("GC=F", {}).get("price"),
        "oil":              macro_raw.get("CL=F", {}).get("price"),
        "bitcoin":          macro_raw.get("BTC-USD", {}).get("price"),
        "sectors":          sectors_raw,
        "hot_sectors":      hot_sectors,
        "cold_sectors":     cold_sectors,
        "qqq_change":       macro_raw.get("QQQ", {}).get("change_pct", 0),
    }


def _fetch_macro() -> dict:
    """Fetch macro indicators (blocking, run in thread pool)."""
    result: dict[str, dict] = {}
    for sym in _MACRO:
        try:
            info = yf.Ticker(sym).info
            price = (
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or info.get("previousClose", 0)
            )
            prev = info.get("regularMarketPreviousClose") or info.get("previousClose", price)
            chg = ((price - prev) / prev * 100) if prev else 0

            ma200 = None
            if sym == "SPY":
                try:
                    hist = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
                    if not hist.empty and len(hist) >= 50:
                        ma200 = float(hist["Close"].iloc[-200:].mean()) if len(hist) >= 200 else float(hist["Close"].mean())
                except Exception:
                    pass

            result[sym] = {"price": round(float(price), 2), "change_pct": round(chg, 2), "ma200": ma200}
        except Exception as e:
            logger.warning(f"_fetch_macro {sym}: {e}")
    return result


def _fetch_sectors() -> dict:
    """Fetch all 11 sector ETF performance levels (blocking)."""
    result: dict[str, dict] = {}
    for name, etf in SECTOR_ETFS.items():
        try:
            info = yf.Ticker(etf).info
            price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
            prev  = info.get("regularMarketPreviousClose") or info.get("previousClose", price)
            chg   = ((price - prev) / prev * 100) if prev else 0
            result[name] = {"etf": etf, "price": round(float(price), 2), "change_pct": round(chg, 2)}
        except Exception as e:
            logger.warning(f"_fetch_sectors {etf}: {e}")
            result[name] = {"etf": etf, "price": None, "change_pct": None}
    return result


def format_for_prompt(ctx: dict, ticker_sector: str | None = None) -> str:
    """
    Format market context as a compact block for AI prompt injection.
    Optionally highlights the sector context for a specific stock's sector.
    """
    if not ctx:
        return "Market context: unavailable."

    spy_arrow = "▲" if ctx.get("spy_change_pct", 0) >= 0 else "▼"
    vix_arrow = "▲" if ctx.get("vix_change", 0) >= 0 else "▼"

    lines = [
        "=== LIVE MARKET CONTEXT ===",
        f"Regime:      {ctx.get('regime_label', 'Unknown')}",
        f"S&P 500:     {spy_arrow}{abs(ctx.get('spy_change_pct', 0)):.2f}% today  "
        f"(NASDAQ {'+' if ctx.get('qqq_change', 0) >= 0 else ''}{ctx.get('qqq_change', 0):.2f}%)",
        f"VIX:         {ctx.get('vix', 'N/A')} {vix_arrow}{abs(ctx.get('vix_change', 0)):.1f}% today",
        f"Rates:       10Y Yield {ctx.get('rate_environment', 'N/A')}",
        f"Dollar:      {ctx.get('dollar_note', 'N/A')}",
    ]

    if ctx.get("gold"):
        lines.append(f"Gold/Oil:    Gold ${ctx['gold']:.0f}  |  Oil ${ctx.get('oil', 0):.1f}/bbl")

    # Sector rotation
    hot = ctx.get("hot_sectors", [])
    cold = ctx.get("cold_sectors", [])
    if hot:
        lines.append(f"Hot sectors: {', '.join(f'{s} ({c:+.1f}%)' for s, c in hot[:3])}")
    if cold:
        lines.append(f"Weak sectors:{', '.join(f'{s} ({c:+.1f}%)' for s, c in cold[:3])}")

    # Highlight the specific stock's sector if provided
    if ticker_sector and ctx.get("sectors"):
        sec_data = ctx["sectors"].get(ticker_sector)
        if sec_data and sec_data.get("change_pct") is not None:
            arrow = "▲" if sec_data["change_pct"] >= 0 else "▼"
            is_hot = ticker_sector in [s for s, _ in hot[:3]]
            tag = " ← this stock's sector is LEADING today" if is_hot else (
                " ← this stock's sector is LAGGING today" if ticker_sector in [s for s, _ in cold[:3]] else ""
            )
            lines.append(
                f"This sector: {ticker_sector} ETF {arrow}{abs(sec_data['change_pct']):.2f}% today{tag}"
            )

    lines.append("===========================")
    return "\n".join(lines)
