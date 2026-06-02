import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import joblib
import os
import numpy as np

# Set page config for mobile-friendly view
st.set_page_config(
    page_title="AI Trade Advisor", 
    page_icon="📈", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e1e2f;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
    }
    .chat-bubble {
        background-color: #2a2a3d;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 8px;
        font-size: 16px;
        line-height: 1.5;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Global Setup & NLP Caching
@st.cache_resource
def load_sentiment_model():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_resource
def load_trade_model():
    model_path = "trade_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

try:
    with st.spinner("Loading AI Models..."):
        sentiment_pipeline = load_sentiment_model()
        trade_model = load_trade_model()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

if trade_model is None:
    st.error("⚠️ 'trade_model.pkl' not found. Please train the offline engine first.")
    st.stop()

st.markdown("<h2 style='text-align: center;'>🤖 AI Trading Assistant</h2>", unsafe_allow_html=True)

# 2. UI Structure (Two Tabs)
tab1, tab2 = st.tabs(["💬 AI Advisor (Hold/Exit)", "🔎 Market Screener (Find Setups)"])

# Helper function to compute features
def compute_features(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    core_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in core_cols:
        if col not in df.columns:
            return None
            
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
    
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14
    ).volume_weighted_average_price()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['Close'], volume=df['Volume']
    ).on_balance_volume()
    df['VROC'] = df['Volume'].pct_change(periods=10) * 100
    
    return df.dropna()

# 3. Tab 1: AI Advisor
with tab1:
    ticker_input = st.text_input("Stock Ticker", value="RELIANCE.NS", placeholder="e.g. RELIANCE.NS", key="t1").strip().upper()
    entry_price = st.number_input("Your Entry Price (Optional)", min_value=0.0, value=0.0, step=1.0)
    advise_btn = st.button("Get AI Advice")
    
    if advise_btn and ticker_input:
        with st.spinner(f"Analyzing {ticker_input}..."):
            try:
                # Fetch 90 days data
                df = yf.download(ticker_input, period="90d", interval="1d", progress=False)
                if df.empty:
                    st.error("No data found for this ticker.")
                else:
                    df_features = compute_features(df)
                    if df_features is None or df_features.empty:
                        st.error("Not enough data to calculate technicals.")
                    else:
                        latest_row = df_features.iloc[[-1]]
                        metrics = df_features.iloc[-1]
                        
                        live_price = metrics['Close']
                        current_rsi = metrics['RSI']
                        current_atr = metrics['ATR']
                        
                        # XGBoost Inference
                        X_input = latest_row
                        if hasattr(trade_model, "feature_names_in_"):
                            X_input = latest_row[trade_model.feature_names_in_]
                        
                        probas = trade_model.predict_proba(X_input)[0]
                        tech_prob = probas[1] * 100
                        
                        # Fetch Fundamentals
                        eps = "N/A"
                        try:
                            ticker_obj = yf.Ticker(ticker_input)
                            info = ticker_obj.info
                            eps = info.get('trailingEPS', 'N/A')
                        except Exception:
                            pass
                            
                        # Fetch News & Sentiment
                        overall_sentiment = "Neutral (No News)"
                        try:
                            news = ticker_obj.news
                            titles = [item['title'] for item in news[:5]] if news else []
                            if titles:
                                sentiments = sentiment_pipeline(titles)
                                pos = sum(1 for s in sentiments if s['label'] == 'positive')
                                neg = sum(1 for s in sentiments if s['label'] == 'negative')
                                if pos > neg:
                                    overall_sentiment = "Positive 📈"
                                elif neg > pos:
                                    overall_sentiment = "Negative 📉"
                                else:
                                    overall_sentiment = "Neutral ⚖️"
                        except Exception:
                            overall_sentiment = "Unknown (API Error)"
                            
                        # Synthesis Logic
                        roi = 0.0
                        if entry_price > 0:
                            roi = ((live_price / entry_price) - 1) * 100
                            roi_text = f"You are currently **up {roi:.2f}%**." if roi > 0 else f"You are currently **down {abs(roi):.2f}%**." if roi < 0 else "You are at breakeven."
                        else:
                            roi_text = "No entry price provided."
                            
                        trend_text = "Bullish" if tech_prob >= 60 else "Bearish" if tech_prob <= 40 else "Neutral"
                        
                        # Conclusion logic
                        if entry_price > 0 and roi > 0:
                            if trend_text == "Bullish":
                                conclusion = f"Hold for more gains. Consider a trailing stop at **{live_price - 2*current_atr:.2f}** to lock in profits."
                            elif trend_text == "Bearish":
                                conclusion = "Momentum is weakening. Exit now to book profits."
                            else:
                                conclusion = f"Consolidating. Tighten stop loss to **{live_price - 1.5*current_atr:.2f}**."
                        elif entry_price > 0 and roi <= 0:
                            if trend_text == "Bullish":
                                conclusion = "Hold. The technicals suggest a potential recovery."
                            else:
                                conclusion = "Consider exiting to cut losses. Trend is not in your favor."
                        else:
                            if trend_text == "Bullish":
                                conclusion = "Good potential entry if seeking new positions. Set stop loss carefully."
                            elif trend_text == "Bearish":
                                conclusion = "Avoid entering new positions right now."
                            else:
                                conclusion = "Wait for a clearer breakout signal."
                                
                        ai_response = (f"{roi_text} The technicals show a **{trend_text}** trend "
                                       f"(Probability: **{tech_prob:.1f}%**). Live news sentiment is **{overall_sentiment}**. "
                                       f"Trailing EPS is **{eps}**.\n\n"
                                       f"**🤖 AI Conclusion:** {conclusion}")
                        
                        # UI Output
                        st.markdown(f"<div class='chat-bubble'>{ai_response}</div>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Live Price", f"{live_price:.2f}")
                        with col2:
                            st.metric("Tech Prob", f"{tech_prob:.1f}%")
                        with col3:
                            st.metric("RSI", f"{current_rsi:.1f}")

            except Exception as e:
                st.error(f"Error during analysis: {e}")

# 4. Tab 2: Market Screener
with tab2:
    st.markdown("### 🔎 Breakout Setup Finder")
    st.write("Scans top liquidity stocks for high upward probability and ideal RSI.")
    
    # Predefined high-liquidity Indian stocks
    default_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
        "INFY.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", 
        "BAJFINANCE.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "HINDUNILVR.NS", "M&M.NS", "SUNPHARMA.NS"
    ]
    
    if st.button("Run Screener (15 Stocks)"):
        screener_data = []
        progress_bar = st.progress(0)
        
        for i, tck in enumerate(default_tickers):
            progress_bar.progress((i + 1) / len(default_tickers))
            try:
                df_s = yf.download(tck, period="90d", interval="1d", progress=False)
                if df_s.empty:
                    continue
                    
                df_feat = compute_features(df_s)
                if df_feat is None or df_feat.empty:
                    continue
                    
                latest = df_feat.iloc[[-1]]
                rsi_val = df_feat.iloc[-1]['RSI']
                close_price = df_feat.iloc[-1]['Close']
                
                X_in = latest
                if hasattr(trade_model, "feature_names_in_"):
                    X_in = latest[trade_model.feature_names_in_]
                    
                prob = trade_model.predict_proba(X_in)[0][1] * 100
                
                screener_data.append({
                    "Ticker": tck,
                    "Price": round(close_price, 2),
                    "Upward Prob (%)": round(prob, 1),
                    "RSI": round(rsi_val, 1)
                })
            except Exception:
                pass
                
        if screener_data:
            df_screen = pd.DataFrame(screener_data).sort_values(by="Upward Prob (%)", ascending=False).reset_index(drop=True)
            
            def highlight_breakout(row):
                # Highlight if >65% prob and RSI between 40-60
                if row['Upward Prob (%)'] > 65 and 40 <= row['RSI'] <= 60:
                    return ['background-color: rgba(0, 200, 83, 0.3)'] * len(row)
                return [''] * len(row)
                
            st.dataframe(df_screen.style.apply(highlight_breakout, axis=1), use_container_width=True)
            
            st.info("💡 **Highlighted rows** indicate >65% Upward Probability with RSI 40-60 (Ideal Breakout Zone).")
            st.warning("⚠️ **Disclaimer:** Targeting 15% in 2-6 weeks requires high volatility and strict risk management. Always use stop losses.")
        else:
            st.error("Screener could not fetch data.")
