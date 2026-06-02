from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
import yfinance as yf
import ta
import xgboost as xgb
import joblib
from transformers import pipeline
import pandas as pd
import numpy as np
import re
import os
import asyncio
import random

# In-memory thread-safe cache for scanner results
LIVE_SCANNER_CACHE = []
SCANNER_LOCK = asyncio.Lock()

# Load ML Models
MODEL_PATH = "../trade_model.pkl" # Root level
try:
    trade_model = joblib.load(MODEL_PATH)
except Exception as e:
    trade_model = None
    print(f"Warning: Offline engine model not found at {MODEL_PATH}. {e}")

try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    sentiment_pipeline = None
    print(f"Warning: FinBERT model failed to load. {e}")

# Hardcoded list of liquid NSE stocks for Macro Filter
NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
    "SUZLON.NS", "TATASTEEL.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "LT.NS", "ITC.NS", "HINDUNILVR.NS", "BAJFINANCE.NS",
    "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS"
]

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
    df['VROC'] = df['Volume'].pct_change(periods=10) * 100
    df.dropna(inplace=True)
    return df

async def macro_filter_loop():
    """Background task to scan NSE stocks every 15 minutes."""
    while True:
        print("Starting Macro Filter Scan...")
        new_candidates = []
        try:
            # Batch fetch via yfinance
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
                    
                    # Positive price acceleration check (Close > EMA 20)
                    if not (40 <= rsi <= 65) or latest['Close'] < latest['EMA_20']:
                        continue
                        
                    tech_prob = 0.0
                    if trade_model is not None:
                        latest_row = latest.to_frame().T
                        X_input = latest_row[trade_model.feature_names_in_] if hasattr(trade_model, "feature_names_in_") else latest_row
                        probas = trade_model.predict_proba(X_input)[0]
                        tech_prob = probas[1]
                        
                    # Filter matching our high-probability metric
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
                except Exception as e:
                    pass # Skip problematic tickers
                    
            # Sort by highest probability
            new_candidates = sorted(new_candidates, key=lambda x: x['prob'], reverse=True)[:20]
            
            async with SCANNER_LOCK:
                global LIVE_SCANNER_CACHE
                LIVE_SCANNER_CACHE = new_candidates
            print(f"Macro Filter Scan Complete. Found {len(new_candidates)} candidates.")
            
        except Exception as e:
            print(f"Macro Filter Error: {e}")
            
        await asyncio.sleep(15 * 60) # Sleep for 15 minutes

async def micro_sniper_loop():
    """Simulates live WebSocket broker updates every 5 seconds for top candidates."""
    while True:
        try:
            async with SCANNER_LOCK:
                for candidate in LIVE_SCANNER_CACHE:
                    # Slightly mutate the LTP by +/- 0.1%
                    mutation = random.uniform(-0.001, 0.001)
                    candidate['price'] = round(candidate['price'] * (1 + mutation), 2)
                    # Dynamically update trailing stop based on new price
                    candidate['stop_loss'] = round(candidate['price'] - 1.5 * candidate['atr'], 2)
                    candidate['target'] = round(candidate['price'] + 2 * candidate['atr'], 2)
        except Exception as e:
            print(f"Micro Sniper Error: {e}")
        await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: spawn background tasks
    task1 = asyncio.create_task(macro_filter_loop())
    task2 = asyncio.create_task(micro_sniper_loop())
    yield
    # Shutdown: cancel tasks
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

def extract_ticker(text: str) -> str:
    words = re.findall(r'\b[A-Z]{3,15}\b', text)
    if words:
        return words[0] + ".NS"
    return ""

def determine_intent(text: str) -> str:
    text = text.lower()
    if any(word in text for word in ["buy", "long", "enter", "setups", "top", "what to buy"]):
        return "setups" if any(w in text for w in ["setups", "top", "what to buy"]) else "buy"
    if any(word in text for word in ["sell", "short", "exit"]):
        return "sell"
    if any(word in text for word in ["hold", "wait", "keep"]):
        return "hold"
    return "analyze"

def get_news_sentiment(ticker: str):
    if sentiment_pipeline is None:
        return "Sentiment engine offline", 0.0
    ticker_obj = yf.Ticker(ticker)
    news = ticker_obj.news
    titles = [item['title'] for item in news[:5]] if news else []
    if not titles:
        return "No recent news found", 0.0
    sentiments = sentiment_pipeline(titles)
    pos = sum(1 for s in sentiments if s['label'] == 'positive')
    neg = sum(1 for s in sentiments if s['label'] == 'negative')
    if pos > neg:
        return "Positive", pos/len(titles)
    elif neg > pos:
        return "Negative", -neg/len(titles)
    return "Neutral", 0.0

@app.get("/api/scanner/results")
async def scanner_endpoint():
    """Returns the live mutating cache of breakout candidates."""
    async with SCANNER_LOCK:
        return {"candidates": LIVE_SCANNER_CACHE}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    message = request.message.lower()
    
    # Intent Router Trigger Keywords
    global_keywords = ["suggest", "top", "find", "best", "portfolio", "give me", "screener"]
    
    # PATH A: Global Recommendation Scan (Zero API Calls)
    if any(kw in message for kw in global_keywords):
        async with SCANNER_LOCK:
            # Sort the cached items by prob descending
            sorted_cache = sorted(LIVE_SCANNER_CACHE, key=lambda x: x['prob'], reverse=True)
            top_candidates = sorted_cache[:10] # Top 5 to 15
            
        if not top_candidates:
            return {"response": "The Macro Filter is still scanning the markets. Please check back in a few minutes."}
            
        resp = "Here are the top algorithmic setups currently cached:\n\n"
        for i, item in enumerate(top_candidates, 1):
            resp += f"**{i}. {item['ticker']}** (Upward Prob: {item['prob']}%) | RSI: {item['rsi']:.1f} | Target: ₹{item['target']:.2f}\n"
        resp += "\n*Disclaimer: These are mathematically filtered setups intended for a 2-6 week horizon.*"
        return {"response": resp}
        
    # PATH B: Specific Asset Lookup (On-Demand API Call)
    ticker = extract_ticker(request.message)
    if not ticker:
        return {"response": "Please specify a valid stock ticker in ALL CAPS (e.g., RELIANCE) so I can fetch live details."}
        
    # Standard analysis (On-Demand API Call)
    df = yf.download(ticker, period="60d", interval="1d", progress=False)
    if df.empty:
        return {"response": f"Could not retrieve data for {ticker}."}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = compute_features(df)
    if df.empty:
        return {"response": f"Could not retrieve enough technical data for {ticker}."}
        
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
            
    sentiment_text, sentiment_score = get_news_sentiment(ticker)
    
    if tech_prob > 60 and sentiment_score >= 0:
        conclusion = f"The technicals are strong (Upward Prob: {tech_prob:.1f}%) and news sentiment is {sentiment_text}. Consider BUYING or holding. A tight trailing stop at ₹{live_price - 2*atr:.2f} is recommended."
    elif tech_prob < 40 or sentiment_score < 0:
        conclusion = f"Caution: Upward Probability is low ({tech_prob:.1f}%) and sentiment is {sentiment_text}. Consider EXITING or taking profits if you are long. Avoid new entries."
    else:
        conclusion = f"The setup is currently mixed (Prob: {tech_prob:.1f}%, Sentiment: {sentiment_text}). I recommend HOLDING or monitoring until a clearer breakout occurs."
        
    response = (
        f"**Analysis for {ticker}:**\n\n"
        f"📈 **Live Price:** ₹{live_price:.2f}\n"
        f"📊 **RSI (14):** {rsi:.1f}\n"
        f"🛡️ **ATR Stop Ref:** ₹{live_price - 2*atr:.2f}\n"
        f"📰 **Live News Sentiment:** {sentiment_text}\n\n"
        f"🤖 **AI Conclusion:** {conclusion}"
    )
    return {"response": response}

# Mount static files at the end to not shadow /api endpoints
app.mount("/", StaticFiles(directory="../", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
