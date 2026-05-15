"""
MON100 ETF Tracker — GitHub Actions Version
============================================
Runs once, sends Telegram signal, exits.
Telegram credentials are read from environment variables (GitHub Secrets).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# CONFIG — credentials come from GitHub Secrets (never hardcode)
# ─────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CAPITAL          = 100_000      # Your capital in Rs — change this

TICKER_MON100 = "MON100.NS"
TICKER_NASDAQ = "^NDX"
TICKER_USDINR = "USDINR=X"
PERIOD        = "6mo"
INTERVAL      = "1d"


# ─────────────────────────────────────────────────────────
# 1. TELEGRAM
# ─────────────────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"[OK] Telegram message sent.")
            return True
        else:
            print(f"[FAIL] Telegram error: {resp.text}")
            return False
    except Exception as e:
        print(f"[FAIL] Telegram exception: {e}")
        return False


# ─────────────────────────────────────────────────────────
# 2. FETCH DATA
# ─────────────────────────────────────────────────────────
def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────
# 3. TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].squeeze()

    df["SMA_20"]  = close.rolling(20).mean()
    df["SMA_50"]  = close.rolling(50).mean()
    df["EMA_9"]   = close.ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma20

    rolling_high = close.rolling(252).max()
    rolling_low  = close.rolling(252).min()
    df["Pct_from_High"] = (close - rolling_high) / rolling_high * 100
    df["Pct_from_Low"]  = (close - rolling_low)  / rolling_low  * 100

    return df


# ─────────────────────────────────────────────────────────
# 4. BUY SIGNAL SCORING
# ─────────────────────────────────────────────────────────
def evaluate_buy_signal(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    close  = float(latest["Close"])
    rsi    = float(latest["RSI"])

    signals = {}
    score   = 0

    checks = [
        ("Price > SMA20",             close > float(latest["SMA_20"])),
        ("Price > SMA50",             close > float(latest["SMA_50"])),
        ("SMA20 > SMA50 (GoldenX)",   float(latest["SMA_20"]) > float(latest["SMA_50"])),
        ("EMA9 rising",               float(latest["EMA_9"]) > float(prev["EMA_9"])),
        ("RSI 40-65 (healthy zone)",  40 <= rsi <= 65),
        ("RSI > 50 (bullish)",        rsi > 50),
        ("MACD Hist +ve and rising",  float(latest["MACD_Hist"]) > 0 and
                                      float(latest["MACD_Hist"]) > float(prev["MACD_Hist"])),
        ("Price below BB Upper",      close <= float(latest["BB_Upper"]) * 0.98),
        ("Within 20% of 52w Low",     float(latest["Pct_from_Low"]) < 20),
        ("BB Width OK",               float(latest["BB_Width"]) > 0.04),
    ]

    for label, cond in checks:
        signals[label] = cond
        if cond:
            score += 1

    if score >= 7:
        verdict, emoji = "STRONG BUY",      "GREEN"
    elif score >= 5:
        verdict, emoji = "MODERATE BUY",    "YELLOW"
    elif score >= 3:
        verdict, emoji = "NEUTRAL / WATCH", "BLUE"
    else:
        verdict, emoji = "AVOID / WAIT",    "RED"

    return {
        "score": score, "verdict": verdict, "emoji": emoji,
        "signals": signals, "close": close, "rsi": rsi,
        "latest": latest,
    }


# ─────────────────────────────────────────────────────────
# 5. POSITION SIZE
# ─────────────────────────────────────────────────────────
def suggest_position(score: int, capital: float) -> dict:
    if score >= 7:
        pct, label = 0.20, "Full position (20%)"
    elif score >= 5:
        pct, label = 0.10, "Half position (10%)"
    elif score >= 3:
        pct, label = 0.05, "SIP / small add (5%)"
    else:
        pct, label = 0.0,  "No new position"
    return {"pct": pct * 100, "amount": capital * pct, "label": label}


# ─────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────
def build_and_send():
    now_ist = datetime.utcnow().strftime("%d %b %Y")
    print(f"[->] Running MON100 analysis for {now_ist}...")

    try:
        df_mon    = fetch_data(TICKER_MON100)
        df_ndx    = fetch_data(TICKER_NASDAQ)
        df_usdinr = fetch_data(TICKER_USDINR)
    except Exception as e:
        send_telegram(f"MON100 Tracker: Data fetch failed\n<code>{e}</code>")
        return

    if df_mon.empty:
        send_telegram("MON100 Tracker: No data today (market holiday?)")
        return

    df_mon  = add_indicators(df_mon)
    result  = evaluate_buy_signal(df_mon)
    pos     = suggest_position(result["score"], CAPITAL)
    latest  = result["latest"]

    mon_price = result["close"]
    mon_prev  = float(df_mon["Close"].iloc[-2])
    ndx_price = float(df_ndx["Close"].iloc[-1]) if not df_ndx.empty else 0
    ndx_prev  = float(df_ndx["Close"].iloc[-2]) if len(df_ndx) > 1 else ndx_price
    usdinr    = float(df_usdinr["Close"].iloc[-1]) if not df_usdinr.empty else 0

    mon_chg = (mon_price - mon_prev) / mon_prev * 100
    ndx_chg = (ndx_price - ndx_prev) / ndx_prev * 100 if ndx_prev else 0

    r1m = None
    if len(df_mon) >= 21:
        r1m = (float(df_mon["Close"].iloc[-1]) / float(df_mon["Close"].iloc[-21]) - 1) * 100

    sl   = mon_price * 0.95
    tgt1 = mon_price * 1.08
    tgt2 = mon_price * 1.15

    checklist = ""
    for label, passed in result["signals"].items():
        icon = "YES" if passed else "NO"
        checklist += f"  [{icon}] {label}\n"

    units_line = ""
    if pos["amount"] > 0:
        units = int(pos["amount"] / mon_price)
        units_line = f"  ~Units : <b>{units} units</b>\n"

    r1m_line = f"  1-Month Return : <b>{r1m:+.2f}%</b>\n" if r1m else ""

    message = (
        f"📊 <b>MON100 DAILY SIGNAL</b>\n"
        f"<i>{now_ist}</i>\n"
        f"{'─'*30}\n\n"
        f"💰 <b>PRICES</b>\n"
        f"  MON100     : <b>Rs.{mon_price:.2f}</b>  ({mon_chg:+.2f}%)\n"
        f"  NASDAQ-100 : <b>${ndx_price:,.2f}</b>  ({ndx_chg:+.2f}%)\n"
        f"  USD/INR    : <b>Rs.{usdinr:.2f}</b>\n"
        f"{r1m_line}\n"
        f"📈 <b>INDICATORS</b>\n"
        f"  RSI (14)   : <b>{result['rsi']:.1f}</b>\n"
        f"  SMA-20     : Rs.{float(latest['SMA_20']):.2f}\n"
        f"  SMA-50     : Rs.{float(latest['SMA_50']):.2f}\n"
        f"  MACD Hist  : {float(latest['MACD_Hist']):.4f}\n"
        f"  BB Upper   : Rs.{float(latest['BB_Upper']):.2f}\n"
        f"  BB Lower   : Rs.{float(latest['BB_Lower']):.2f}\n\n"
        f"🔍 <b>SIGNAL CHECKLIST</b>\n"
        f"{checklist}\n"
        f"{'─'*30}\n"
        f"<b>SCORE: {result['score']}/10 — {result['verdict']}</b>\n"
        f"{'─'*30}\n\n"
        f"💼 <b>POSITION (Capital Rs.{CAPITAL:,.0f})</b>\n"
        f"  Strategy : {pos['label']}\n"
        f"  Amount   : <b>Rs.{pos['amount']:,.0f}</b>\n"
        f"{units_line}\n"
        f"🎯 <b>LEVELS (if BUY)</b>\n"
        f"  Stop-Loss : Rs.{sl:.2f}  (-5%)\n"
        f"  Target 1  : Rs.{tgt1:.2f}  (+8%)\n"
        f"  Target 2  : Rs.{tgt2:.2f}  (+15%)\n\n"
        f"<i>Educational only. Not financial advice.</i>"
    )

    send_telegram(message)
    print(message)


if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set.")
        exit(1)
    build_and_send()