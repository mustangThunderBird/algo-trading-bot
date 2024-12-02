import os
import logging
import pickle
import pandas as pd
import yfinance as yf
from quantitative.quant_model import preprocess_ticker_data

# Suppress TensorFlow logs
import absl.logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)

def get_recent_data(ticker, period="1mo"):
    """Fetch the last 30 days of stock data for the given ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return hist
    except Exception as e:
        raise RuntimeError(f"Error fetching data for {ticker}: {e}")
    
def preprocess_for_prediction(recent_data):
    """Preprocess the fetched data to create features for the model."""
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in recent_data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in stock data: {missing_cols}")

    processed_data = preprocess_ticker_data(recent_data)

    feature_columns = ['Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag4',
                       'ROC_5', 'MA_Return_5', 'Volatility_5', 'Volatility_10',
                       'RSI', 'OBV', 'MACD', 'MACD_Signal']
    latest_features = processed_data[feature_columns].iloc[-1]

    return pd.DataFrame([latest_features], columns=feature_columns)  # Return as DataFrame

def predict_ticker(ticker, model):
    """Fetch, preprocess, and predict for a single ticker."""
    try:
        recent_data = get_recent_data(ticker)
        input_features = preprocess_for_prediction(recent_data)
        prediction = model.predict(input_features)[0]
        return prediction
    except Exception as e:
        logging.error(f"Error predicting for {ticker}: {e}")
        return None