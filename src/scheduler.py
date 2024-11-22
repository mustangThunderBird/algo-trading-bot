from apscheduler.schedulers.background import BackgroundScheduler
import time
import os
import logging

QUALITATIVE_MODEL_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'model', 'qualitative', 'qual_model.py')
QUANTITATIVE_MODEL_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'model', 'quantitative', 'batch_train.py')

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

def run_qualitative_model():
    '''
    Function to run the qual model
    and get new sentiment scores
    '''
    try:
        logging.info("Starting qual model execution...")
        #qual_model.determine_sentiments()
        os.system(f"python3 {QUALITATIVE_MODEL_SCRIPT_PATH}")
        logging.info("Qualitative model execution completed successfully")
    except Exception as e:
        logging.error(f"Error running qual model: {e}")

def run_quantitative_model():
    '''
    Function to run the quantitative model
    and get new real stock data
    '''
    try:
        logging.info("Starting quant model execution...")
        #batch_train.train_models(pull_data=True)
        os.system(f"python3 {QUANTITATIVE_MODEL_SCRIPT_PATH}")
        logging.info("Quantitative model execution completed successfully")
    except Exception as e:
        logging.error(f"Error running quant model: {e}")

def main():
    scheduler = BackgroundScheduler()

    scheduler.add_job(run_qualitative_model, 'cron', day_of_week="mon-fri", hour=4)
    scheduler.add_job(run_quantitative_model, 'cron', day_of_week='sat', hour=10)

    scheduler.start()
    logging.info("Scheduler started successfully. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except (KeyboardInterrupt, SystemExit):
        logging.info("Shutting down scheduler...")
        scheduler.shutdown()

if __name__ == "__main__":
    main()