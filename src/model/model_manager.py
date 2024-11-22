import pandas as pd
import os
import pickle
import logging
import model_handler as mh

# Ensure logs directory exists
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Clear the log file
with open(os.path.join(LOG_DIR, 'model_manager.log'), 'w') as log_file:
    log_file.write("") 

# Configure logging to write to a file
LOG_FILE = os.path.join(LOG_DIR, 'model_manager.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelManager:
    def __init__(self, sentiment_file, quant_model_dir, quant_weight=0.8, qual_weight=0.2):
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
        logging.info(f"Loaded sentiment scores for {len(sentiments)} tickers. Index type: {sentiments.index.dtype}")
        return sentiments


    def normalize(self, values):
        """Normalize a series of values to [0, 1]."""
        min_val = values.min()
        max_val = values.max()
        return (values - min_val) / (max_val - min_val)

    def make_decisions(self, output_file):
        """Make buy/sell/hold decisions and save them incrementally."""
        with open(output_file, 'w') as f:
            # Write header
            f.write("ticker,next_day_return,sentiment_score,decision_score,action\n")
            
            # Process models one by one
            for ticker, model in self.model_generator():
                ticker = str(ticker)
                logging.info(f"Processing model for ticker: {ticker}")
                try:
                    # Predict next-day return
                    next_day_return = mh.predict_ticker(ticker, model)
                    logging.info(f"Predicted next day return for {ticker} is {next_day_return*100}%")
                except Exception as e:
                    logging.error(f"Error predicting for {ticker}: {e}")
                    continue

                # Get sentiment score
                try:
                    sentiment_score = self.sentiments.loc[ticker, 'sentiment_score']
                    logging.info(f"News sentiment score for {ticker} is {sentiment_score}")
                except KeyError:
                    logging.warning(f"No sentiment score found for {ticker}. Skipping.")
                    continue

                # Normalize return and sentiment scores
                normalized_return = self.normalize(pd.Series([next_day_return]))[0]
                normalized_sentiment = (sentiment_score + 1) / 2
                logging.info(f"Normalized scores for {ticker} are {normalized_return*100}% and {sentiment_score}")

                # Compute decision score
                decision_score = (
                    self.quant_weight * normalized_return +
                    self.qual_weight * normalized_sentiment
                )
                logging.info(f"Decision score for {ticker} is {decision_score}")

                # Determine action
                if decision_score > 0.6:
                    action = "Buy"
                elif decision_score < 0.4:
                    action = "Sell"
                else:
                    action = "Hold"

                # Write decision to file
                f.write(f"{ticker},{next_day_return},{sentiment_score},{decision_score},{action}\n")

try:
    sentiment_file = os.path.join(os.path.dirname(__file__), 'qualitative', 'sentiment_scores.csv')
    model_dir = os.path.join(os.path.dirname(__file__), 'quantitative', 'models')
    model_manager = ModelManager(sentiment_file, model_dir)

    decisions_file = os.path.join(LOG_DIR, 'buy_sell_decisions.csv')
    model_manager.make_decisions(decisions_file)
    logging.info(f"Decisions saved to {decisions_file}")
except Exception as e:
    logging.error(f"Error in decision making: {e}")
