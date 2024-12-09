from apscheduler.schedulers.background import BackgroundScheduler
from model.qualitative import qual_model
from model.quantitative import batch_train
from model.model_manager import ModelManager
import trade_execution
import os
import logging

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Clear the log file
with open(os.path.join(LOG_DIR, 'scheduler.log'), 'w') as log_file:
    log_file.write("") 

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'scheduler.log'),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Scheduler:
    def __init__(self):
        self.running = False
        self.scheduler = None

    def run_qualitative_model(self):
        '''
        Function to run the qualitative model and get new sentiment scores.
        '''
        try:
            logging.info("Starting qualitative model execution...")
            qual_model.determine_sentiments()
            logging.info("Qualitative model execution completed successfully")
        except Exception as e:
            logging.error(f"Error running qualitative model: {e}")

    def run_quantitative_model(self):
        '''
        Function to run the quantitative model and get new real stock data.
        '''
        try:
            logging.info("Starting quantitative model execution...")
            batch_train.train_models()
            logging.info("Quantitative model execution completed successfully")
        except Exception as e:
            logging.error(f"Error running quantitative model: {e}")

    def run_trade_execution(self):
        '''
        Function to execute trades based on the decisions.
        '''
        try:
            logging.info("Starting trade execution...")
            sentiment_file = os.path.join(os.path.dirname(__file__), 'qualitative', 'sentiment_scores.csv')
            model_dir = os.path.join(os.path.dirname(__file__), 'quantitative', 'models')
            mm = ModelManager(sentiment_file, model_dir)
            decisions_file = os.path.join(LOG_DIR, 'buy_sell_decisions.csv')
            mm.make_decisions(decisions_file)
            trade_execution.execute_trades()  # Call the function directly
            logging.info("Trade execution completed successfully")
        except Exception as e:
            logging.error(f"Error during trade execution: {e}")

    def start(self):
        if self.running:
            print("Scheduler is already running.")
            return

        print("Starting scheduler...")
        self.running = True

        # Initialize the BackgroundScheduler and add jobs
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.run_qualitative_model, 'cron', day_of_week="mon-fri", hour=4)
        self.scheduler.add_job(self.run_quantitative_model, 'cron', day_of_week='sat', hour=10)
        self.scheduler.add_job(self.run_trade_execution, 'cron', hour=9)

        self.scheduler.start()
        logging.info("Scheduler started successfully.")
        print("Scheduler started successfully.")

    def stop(self):
        if not self.running:
            print("Scheduler is not running.")
            return

        print("Stopping scheduler...")
        try:
            self.scheduler.shutdown(wait=True)  # Wait for jobs to finish
            logging.info("Scheduler stopped successfully.")
            print("Scheduler stopped successfully.")
        except Exception as e:
            logging.error(f"Error during scheduler shutdown: {e}")
            print(f"Error stopping scheduler: {e}")
        finally:
            self.running = False