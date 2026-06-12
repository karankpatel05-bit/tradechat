from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
import yfinance as yf
import ta
import xgboost as xgb
import joblib
from google import genai
import pandas as pd
import numpy as np
import re
import os
import asyncio
import time
import random

# ============================================================
# GEMINI AI CONFIGURATION (FREE TIER - 1500 req/day)
# Get your free key at: https://aistudio.google.com/apikey
# Set via: export GEMINI_API_KEY="your-key-here"
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client = None

SYSTEM_PROMPT = (
    "You are an expert Indian Share Market Analyst AI. "
    "Your name is Quant Trade AI. "
    "You specialize in NSE/BSE stocks, IPOs, demergers, mutual funds, "
    "sectoral analysis, and portfolio strategy for Indian retail investors. "
    "Always provide detailed, well-structured answers with bullet points. "
    "Include approximate price levels, brokerage estimates, and risk factors when relevant. "
    "Use ₹ for Indian Rupee. Reference sources like SEBI, NSE, BSE, or major brokerages "
    "when making claims. Keep the tone professional but accessible. "
    "If you don't know something, say so honestly. "
    "Format your responses with **bold** headers and clean structure."
)

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini AI client initialized successfully (Free Tier).")
    except Exception as e:
        gemini_client = None
        print(f"Warning: Gemini AI failed to initialize. {e}")
else:
    print("Warning: GEMINI_API_KEY not set. General market Q&A will be unavailable.")

# ============================================================
# IN-MEMORY CACHE & ML MODEL LOADING
# ============================================================
LIVE_SCANNER_CACHE = []
SCANNER_LOCK = asyncio.Lock()

MODEL_PATH = "../trade_model.pkl"
try:
    trade_model = joblib.load(MODEL_PATH)
except Exception as e:
    trade_model = None
    print(f"Warning: Offline engine model not found at {MODEL_PATH}. {e}")

try:
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except ImportError:
    sentiment_pipeline = None
    print("Info: FinBERT not installed — using Gemini AI for sentiment analysis instead.")
except Exception as e:
    sentiment_pipeline = None
    print(f"Warning: FinBERT model failed to load. {e}")

# ============================================================
# NSE STOCK UNIVERSE FOR MACRO FILTER
# ============================================================
NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SUZLON.NS", "TATASTEEL.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "LT.NS", "ITC.NS", "HINDUNILVR.NS", "BAJFINANCE.NS",
    "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS"
]

# ============================================================
# TECHNICAL FEATURE COMPUTATION
# ============================================================
def compute_features(df):
    core_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in core_cols:
        if col not in df.columns:
            return pd.DataFrame()

    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14
    ).volume_weighted_average_price()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['Close'], volume=df['Volume']
    ).on_balance_volume()
    df['VROC'] = df['Volume'].pct_change(periods=10, fill_method=None) * 100
    df.dropna(inplace=True)
    return df

# ============================================================
# BACKGROUND ASYNC LOOPS
# ============================================================
async def macro_filter_loop():
    """Background task to scan NSE stocks every 15 minutes."""
    while True:
        print("Starting Macro Filter Scan...")
        new_candidates = []
        try:
            data = yf.download(NSE_STOCKS, period="60d", interval="1d", group_by="ticker", progress=False)

            for ticker in NSE_STOCKS:
                try:
                    df = data[ticker].copy() if len(NSE_STOCKS) > 1 else data.copy()
                    if df.empty:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    df = compute_features(df)
                    if df.empty:
                        continue

                    latest = df.iloc[-1]
                    rsi = latest['RSI']

                    if not (40 <= rsi <= 65) or latest['Close'] < latest['EMA_20']:
                        continue

                    tech_prob = 0.0
                    if trade_model is not None:
                        latest_row = latest.to_frame().T
                        X_input = latest_row[trade_model.feature_names_in_] if hasattr(trade_model, "feature_names_in_") else latest_row
                        probas = trade_model.predict_proba(X_input)[0]
                        tech_prob = probas[1]

                    if tech_prob >= 0.58:
                        new_candidates.append({
                            "ticker": ticker,
                            "price": float(latest['Close']),
                            "rsi": float(rsi),
                            "atr": float(latest['ATR']),
                            "prob": round(tech_prob * 100, 1),
                            "target": float(latest['Close'] + 2 * latest['ATR']),
                            "stop_loss": float(latest['Close'] - 1.5 * latest['ATR'])
                        })
                except Exception:
                    pass

            new_candidates = sorted(new_candidates, key=lambda x: x['prob'], reverse=True)[:20]

            async with SCANNER_LOCK:
                global LIVE_SCANNER_CACHE
                LIVE_SCANNER_CACHE = new_candidates
            print(f"Macro Filter Scan Complete. Found {len(new_candidates)} candidates.")

        except Exception as e:
            print(f"Macro Filter Error: {e}")

        await asyncio.sleep(15 * 60)

async def micro_sniper_loop():
    """Simulates live WebSocket broker updates every 5 seconds."""
    while True:
        try:
            async with SCANNER_LOCK:
                for candidate in LIVE_SCANNER_CACHE:
                    mutation = random.uniform(-0.001, 0.001)
                    candidate['price'] = round(candidate['price'] * (1 + mutation), 2)
                    candidate['stop_loss'] = round(candidate['price'] - 1.5 * candidate['atr'], 2)
                    candidate['target'] = round(candidate['price'] + 2 * candidate['atr'], 2)
        except Exception as e:
            print(f"Micro Sniper Error: {e}")
        await asyncio.sleep(5)

# ============================================================
# FASTAPI LIFESPAN & SETUP
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    task1 = asyncio.create_task(macro_filter_loop())
    task2 = asyncio.create_task(micro_sniper_loop())
    yield
    task1.cancel()
    task2.cancel()

app = FastAPI(title="Hybrid Quant AI Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# ============================================================
# HELPER FUNCTIONS
# ============================================================
# Common stock name -> ticker aliases for Indian market
TICKER_ALIASES = {
    # Major stocks
    "hdfc bank": "HDFCBANK", "hdfc": "HDFCBANK", "reliance": "RELIANCE",
    "tcs": "TCS", "infosys": "INFY", "infy": "INFY",
    "icici bank": "ICICIBANK", "icici": "ICICIBANK",
    "sbi": "SBIN", "state bank": "SBIN",
    "tata steel": "TATASTEEL", "suzlon": "SUZLON",
    "bharti airtel": "BHARTIARTL", "airtel": "BHARTIARTL",
    "kotak bank": "KOTAKBANK", "kotak": "KOTAKBANK",
    "axis bank": "AXISBANK", "axis": "AXISBANK",
    "itc": "ITC", "hul": "HINDUNILVR", "hindustan unilever": "HINDUNILVR",
    "bajaj finance": "BAJFINANCE", "bajaj": "BAJFINANCE",
    "asian paints": "ASIANPAINT", "hcl tech": "HCLTECH", "hcltech": "HCLTECH",
    "maruti": "MARUTI", "sun pharma": "SUNPHARMA", "ultratech": "ULTRACEMCO",
    "larsen": "LT", "wipro": "WIPRO", "adani": "ADANIENT",
    "tata motors": "TATAMOTORS", "tata power": "TATAPOWER", "tata elxsi": "TATAELXSI",
    "vedanta": "VEDL", "power grid": "POWERGRID", "ntpc": "NTPC",
    "coal india": "COALINDIA", "ongc": "ONGC", "ioc": "IOC",
    "bpcl": "BPCL", "gail": "GAIL", "jsw steel": "JSWSTEEL",
    "tech mahindra": "TECHM", "mahindra": "M&M",
    "titan": "TITAN", "zomato": "ZOMATO", "paytm": "PAYTM",
    "raymond": "RAYMOND", "dmart": "DMART", "pidilite": "PIDILITIND",
    "indigo": "INDIGO", "irctc": "IRCTC", "jio financial": "JIOFIN",
    "adani green": "ADANIGREEN", "adani ports": "ADANIPORTS",
    "adani enterprises": "ADANIENT", "adani power": "ADANIPOWER",
    "hero moto": "HEROMOTOCO", "eicher": "EICHERMOT", "bajaj auto": "BAJAJ-AUTO",
    "nestle": "NESTLEIND", "britannia": "BRITANNIA", "dabur": "DABUR",
    "godrej": "GODREJCP", "hindustan zinc": "HINDZINC",
    "cipla": "CIPLA", "dr reddy": "DRREDDY", "divis lab": "DIVISLAB",
    "apollo hospital": "APOLLOHOSP", "sbi life": "SBILIFE",
    "hdfc life": "HDFCLIFE", "icici prudential": "ICICIPRULI",
    # Gold & Silver ETFs
    "tata gold": "TATAGOLD", "gold etf": "GOLDBEES", "gold bees": "GOLDBEES",
    "sbi gold": "SETFGOLD", "nippon gold": "GOLDSHARE",
    "silver etf": "SILVERBEES", "silver bees": "SILVERBEES", "tata silver": "TATASILVER",
    # Index ETFs
    "nifty bees": "NIFTYBEES", "nifty etf": "NIFTYBEES",
    "bank bees": "BANKBEES", "junior bees": "JUNIORBEES",
}

def extract_ticker(text: str) -> str:
    """Extract an NSE ticker from user text — case-insensitive with aliases."""
    text_lower = text.lower()
    
    # Check aliases first (handles 'hdfc bank', 'tata gold etf', etc.)
    # Sort by length descending so 'tata gold' matches before 'tata'
    for alias, ticker in sorted(TICKER_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in text_lower:
            return ticker + ".NS"
    
    # Fallback: look for uppercase ticker symbols in original text (e.g. RELIANCE, TCS)
    words = re.findall(r'\b[A-Z]{3,15}\b', text)
    if words:
        return words[0] + ".NS"
    
    return ""

async def get_news_sentiment(ticker: str):
    """Analyze news sentiment using FinBERT (local) or Gemini AI (cloud fallback)."""
    ticker_obj = yf.Ticker(ticker)
    news = ticker_obj.news
    titles = [item['title'] for item in news[:5]] if news else []
    if not titles:
        return "No recent news found", 0.0

    # Try FinBERT first (available when transformers is installed)
    if sentiment_pipeline is not None:
        sentiments = sentiment_pipeline(titles)
        pos = sum(1 for s in sentiments if s['label'] == 'positive')
        neg = sum(1 for s in sentiments if s['label'] == 'negative')
        if pos > neg:
            return "Positive", pos / len(titles)
        elif neg > pos:
            return "Negative", -neg / len(titles)
        return "Neutral", 0.0

    # Fallback: Use Gemini AI for sentiment analysis
    if gemini_client is not None:
        try:
            prompt = (
                f"Analyze the sentiment of these {ticker} news headlines. "
                f"Reply with ONLY one word: Positive, Negative, or Neutral.\n\n"
                + "\n".join(f"- {t}" for t in titles)
            )
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            result = response.text.strip().lower()
            if "positive" in result:
                return "Positive (AI)", 0.5
            elif "negative" in result:
                return "Negative (AI)", -0.5
            return "Neutral (AI)", 0.0
        except Exception as e:
            print(f"Gemini sentiment error: {e}")
            return "Sentiment unavailable", 0.0

    return "Sentiment engine offline", 0.0

async def ask_gemini(user_query: str) -> str:
    """Send a general market question to Google Gemini (Free Tier) with retry logic."""
    if gemini_client is None:
        return (
            "The AI knowledge engine is not configured yet.\n\n"
            "**Setup (Free, 2 minutes):**\n"
            "1. Go to https://aistudio.google.com/apikey\n"
            "2. Click 'Create API Key'\n"
            "3. Set it on your server: export GEMINI_API_KEY=\"your-key\"\n"
            "4. Restart the server"
        )
    
    # Retry with exponential backoff (handles 429 rate limits)
    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_query,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if '429' in error_str and attempt < 2:
                wait_time = (attempt + 1) * 15  # 15s, 30s
                print(f"Gemini rate limited. Retrying in {wait_time}s (attempt {attempt + 1}/3)")
                await asyncio.sleep(wait_time)
                continue
            print(f"Gemini API Error: {e}")
            return (
                "I'm currently rate-limited by the AI engine. Please try again in a minute.\n\n"
                "**Tip:** You can still ask about specific stocks! Try typing a ticker like:\n"
                "• RELIANCE\n• HDFC Bank\n• TCS\n• TATASTEEL"
            )

# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/api/scanner/results")
async def scanner_endpoint():
    """Returns the live mutating cache of breakout candidates."""
    async with SCANNER_LOCK:
        return {"candidates": LIVE_SCANNER_CACHE}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Tri-Path Intent Router:
      Path A -> Cached scanner results (zero API calls)
      Path B -> Specific ticker deep-analysis (yfinance + XGBoost + FinBERT)
      Path C -> General market Q&A via Gemini AI (IPOs, demergers, news, etc.)
    """
    message = request.message
    message_lower = message.lower()

    # ---------------------------------------------------------
    # PATH A: Global Recommendation Scan (Zero API Calls)
    # ---------------------------------------------------------
    global_keywords = ["suggest", "top", "find", "best", "portfolio", "give me", "screener", "setups"]

    if any(kw in message_lower for kw in global_keywords):
        async with SCANNER_LOCK:
            sorted_cache = sorted(LIVE_SCANNER_CACHE, key=lambda x: x['prob'], reverse=True)
            top_candidates = sorted_cache[:10]

        if not top_candidates:
            return {"response": "The Macro Filter is still scanning the markets. Please check back in a few minutes."}

        resp = "**📊 Top Algorithmic Setups (Live Cache):**\n\n"
        for i, item in enumerate(top_candidates, 1):
            resp += f"**{i}. {item['ticker']}** — Prob: {item['prob']}% | RSI: {item['rsi']:.1f} | LTP: ₹{item['price']:.2f} | Target: ₹{item['target']:.2f}\n"
        resp += "\n*Disclaimer: Mathematically filtered setups for a 2-6 week horizon. Not investment advice.*"
        return {"response": resp}

    # ---------------------------------------------------------
    # PATH B: Specific Asset Lookup (On-Demand yfinance Call)
    # ---------------------------------------------------------
    ticker = extract_ticker(message)

    if ticker:
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df.empty:
            # Ticker not found on yfinance — fall through to Gemini for a smart answer
            gemini_response = await ask_gemini(message)
            return {"response": gemini_response}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = compute_features(df)
        if df.empty:
            gemini_response = await ask_gemini(message)
            return {"response": gemini_response}

        latest = df.iloc[-1]
        live_price = float(latest['Close'])
        rsi = float(latest['RSI'])
        atr = float(latest['ATR'])

        tech_prob = 50.0
        if trade_model is not None:
            try:
                latest_row = latest.to_frame().T
                X_input = latest_row[trade_model.feature_names_in_] if hasattr(trade_model, "feature_names_in_") else latest_row
                probas = trade_model.predict_proba(X_input)[0]
                tech_prob = probas[1] * 100
            except Exception as e:
                print(f"Prediction Error: {e}")

        sentiment_text, sentiment_score = await get_news_sentiment(ticker)

        if tech_prob > 60 and sentiment_score >= 0:
            conclusion = (
                f"Technicals are strong (Upward Prob: {tech_prob:.1f}%) and sentiment is {sentiment_text}. "
                f"Consider BUYING or holding. Trailing stop at ₹{live_price - 2 * atr:.2f}."
            )
        elif tech_prob < 40 or sentiment_score < 0:
            conclusion = (
                f"Caution: Upward Probability is low ({tech_prob:.1f}%) and sentiment is {sentiment_text}. "
                f"Consider EXITING or taking profits. Avoid new entries."
            )
        else:
            conclusion = (
                f"Mixed setup (Prob: {tech_prob:.1f}%, Sentiment: {sentiment_text}). "
                f"HOLD or monitor until a clearer breakout occurs."
            )

        response = (
            f"**Analysis for {ticker}:**\n\n"
            f"📈 **Live Price:** ₹{live_price:.2f}\n"
            f"📊 **RSI (14):** {rsi:.1f}\n"
            f"🛡️ **ATR Stop Ref:** ₹{live_price - 2 * atr:.2f}\n"
            f"📰 **Live News Sentiment:** {sentiment_text}\n\n"
            f"🤖 **AI Conclusion:** {conclusion}"
        )
        return {"response": response}

    # ---------------------------------------------------------
    # PATH C: General Market Q&A via Gemini AI (FREE)
    # Handles: IPOs, demergers, market news, strategy, regulations
    # ---------------------------------------------------------
    gemini_response = await ask_gemini(message)
    return {"response": gemini_response}

# Mount static files at the end to not shadow /api endpoints
app.mount("/", StaticFiles(directory="../", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
