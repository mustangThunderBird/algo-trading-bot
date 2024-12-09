import pandas as pd
import os
import pickle
import logging
#import model_handler as mh
from model import model_handler as mh
import sys

# Ensure logs directory exists
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

class ModelManager:
    def __init__(self, sentiment_file, quant_model_dir, quant_weight=0.85, qual_weight=0.15):
        self.sentiment_file = sentiment_file
        self.quant_model_dir = quant_model_dir
        self.quant_weight = quant_weight
        self.qual_weight = qual_weight
        self.sentiments = self.load_sentiments()

    def model_generator(self):
        """Generator to load models one by one."""
        model_files = [f for f in os.listdir(self.quant_model_dir) if f.endswith('.pkl')]
        for file in model_files:
            ticker = file.split("_")[0]
            with open(os.path.join(self.quant_model_dir, file), 'rb') as f:
                yield ticker, pickle.load(f)

    def load_sentiments(self):
        """Load sentiment scores from the CSV."""
        if not os.path.exists(self.sentiment_file):
            raise FileNotFoundError(f"Sentiment file not found: {self.sentiment_file}")
        
        sentiments = pd.read_csv(self.sentiment_file, index_col=0)  # Use the first column as the index
        sentiments.index = sentiments.index.astype(str)  # Convert the index to strings
        print(f"Loaded sentiment scores for {len(sentiments)} tickers. Index type: {sentiments.index.dtype}")
        return sentiments

    def make_decisions(self, output_file):
        """Make buy/sell/hold decisions and save them incrementally."""
        with open(output_file, 'w') as f:
            # Write header
            f.write("ticker,next_day_return,sentiment_score,decision_score,action\n")
            
            # Process models one by one
            for ticker, model in self.model_generator():
                ticker = str(ticker)
                print(f"Processing model for ticker: {ticker}")
                try:
                    # Predict next-day return
                    next_day_return = mh.predict_ticker(ticker, model)
                    print(f"Predicted next day return for {ticker} is {next_day_return*100}%")
                except Exception as e:
                    logging.error(f"Error predicting for {ticker}: {e}")
                    continue

                # Get sentiment score
                try:
                    sentiment_score = self.sentiments.loc[ticker, 'sentiment_score']
                    print(f"News sentiment score for {ticker} is {sentiment_score}")
                except KeyError:
                    logging.warning(f"No sentiment score found for {ticker}. Skipping.")
                    continue

                # Compute decision score
                decision_score = (
                    self.quant_weight * (next_day_return*100) +
                    self.qual_weight * sentiment_score
                )
                print(f"Decision score for {ticker} is {decision_score}")

                # Determine action
                if decision_score > 1:
                    action = "Buy"
                elif decision_score < 0:
                    action = "Sell"
                else:
                    action = "Hold"

                # Write decision to file
                f.write(f"{ticker},{next_day_return},{sentiment_score},{decision_score},{action}\n")

if __name__ == "__main__" and not hasattr(sys, 'frozen'):
    try:
        sentiment_file = os.path.join(os.path.dirname(__file__), 'qualitative', 'sentiment_scores.csv')
        model_dir = os.path.join(os.path.dirname(__file__), 'quantitative', 'models')
        model_manager = ModelManager(sentiment_file, model_dir)

        decisions_file = os.path.join(LOG_DIR, 'buy_sell_decisions.csv')
        model_manager.make_decisions(decisions_file)
        print(f"Decisions saved to {decisions_file}")
    except Exception as e:
        logging.error(f"Error in decision making: {e}")
