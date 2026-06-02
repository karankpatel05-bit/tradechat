import yfinance as yf
import pandas as pd
import ta
import xgboost as xgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import numpy as np

def train_model(ticker="RELIANCE.NS"):
    print(f"Downloading historical data for {ticker}...")
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    
    if df.empty:
        print("No data found!")
        return

    # Robust fix for pd.MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    core_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in core_cols:
        if col not in df.columns:
            print(f"Missing essential column: {col}")
            return
            
    print("Calculating existing technical indicators...")
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
    
    print("Calculating new volume features...")
    # VWAP requires typical price (high+low+close)/3 and volume
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14
    ).volume_weighted_average_price()
    
    # On-Balance Volume (OBV)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['Close'], volume=df['Volume']
    ).on_balance_volume()
    
    # Volume Rate of Change (VROC 10-period)
    df['VROC'] = df['Volume'].pct_change(periods=10) * 100
    
    print("Defining target (>1% up in 3 days)...")
    # Shift(-3) means looking 3 days into the future.
    # We want future price > current price * 1.01
    future_close = df['Close'].shift(-3)
    df['Target'] = (future_close > (df['Close'] * 1.01)).astype(int)
    
    # Drop rows with NaN (from indicators and shifted target) or Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    features = ['RSI', 'EMA_20', 'EMA_50', 'ATR', 'ADX', 'VWAP', 'OBV', 'VROC', 'Close', 'Volume']
    X = df[features]
    y = df['Target']
    
    print("Running Purged Walk-Forward Validation...")
    # TimeSeriesSplit n_splits=5, test_size=30
    tscv = TimeSeriesSplit(n_splits=5, test_size=30)
    purge_size = 5
    
    auc_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Purge to remove data leakage
        if len(train_idx) > purge_size:
            train_idx_purged = train_idx[:-purge_size]
        else:
            train_idx_purged = train_idx
            
        X_train, y_train = X.iloc[train_idx_purged], y.iloc[train_idx_purged]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        model_cv = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
        model_cv.fit(X_train, y_train)
        
        preds = model_cv.predict_proba(X_test)[:, 1]
        
        # Calculate AUC only if there are both classes in y_test
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, preds)
            auc_scores.append(auc)
            print(f"Fold {fold+1} Validation AUC: {auc:.4f}")
        else:
            print(f"Fold {fold+1} Validation AUC: Skipped (only 1 class in test set)")
            
    if auc_scores:
        print(f"\nAverage Validation AUC: {np.mean(auc_scores):.4f}")
        
    print("\nTraining final model on all available data (purging last 3 unverified targets)...")
    final_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    final_model.fit(X, y)
    
    joblib.dump(final_model, "trade_model.pkl")
    print("Model saved to trade_model.pkl successfully!")

if __name__ == "__main__":
    train_model()
