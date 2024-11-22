from quantitative.quant_model import preprocess_ticker_data
import yfinance as yf
import pandas as pd
import logging
import pickle

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
    # Ensure the data columns match expectations
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in recent_data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in stock data: {missing_cols}")

    # Preprocess data to generate features
    processed_data = preprocess_ticker_data(recent_data)

    # Extract only the latest row of features
    feature_columns = ['Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag4',
                       'ROC_5', 'MA_Return_5', 'Volatility_5', 'Volatility_10',
                       'RSI', 'OBV', 'MACD', 'MACD_Signal']
    latest_features = processed_data[feature_columns].iloc[-1]

    return latest_features.values.reshape(1, -1)  # Return as 2D array for the model

def predict_ticker(ticker, model):
    """Fetch, preprocess, and predict for a single ticker."""
    try:
        # Fetch recent data
        recent_data = get_recent_data(ticker)

        # Preprocess for prediction
        input_features = preprocess_for_prediction(recent_data)

        # Make the prediction
        prediction = model.predict(input_features)[0]

        return prediction
    except Exception as e:
        logging.error(f"Error predicting for {ticker}: {e}")
        return None
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    # Load model
    ticker = "INTC"
    model_path = "/home/crisco/source/repos/gradschool/capstone/algo-trading-bot/src/model/quantitative/models/INTC_quant_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Predict using the full pipeline
    prediction = predict_ticker(ticker, model)
    print(f"Prediction for {ticker}: {prediction}")