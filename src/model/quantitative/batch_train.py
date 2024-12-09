import os
import sys
import matplotlib
import platform
import warnings
from model.quantitative import quant_model as qm
from tensorflow.keras import backend as K
from memory_profiler import profile
import gc

@profile
def batch_train(t, d):
    """
    Train the quantitative model for a given ticker and its data.
    """
    results = qm.build_quant_model(t, d, force_rebuild=True)
    if results:
        print(f"Successfully built model for {t}")
    K.clear_session()
    del results
    

def train_models(pull_data=True, progress_callback=None):
    """
    Train quantitative models for all stocks in the dataset.
    
    Args:
        pull_data (bool): Whether to pull fresh data before training.
        progress_callback (function): A callback function to emit progress updates (percentage).
    """
    if pull_data:
        from model.quantitative.data_download import get_data
        print("Pulling data using data_download.py...")
        get_data()

    warnings.filterwarnings('ignore')
    if os.name == "posix":
        system_name = platform.system()
        if system_name == "Linux":
            os.system("export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")
    matplotlib.use('Agg')

    filepath = os.path.join(os.path.dirname(__file__), 'all_stock_data.csv')
    ticker_data = qm.preprocess_all_stocks_data(filepath=filepath)

    # Initialize progress tracking
    total = len(ticker_data)
    if progress_callback and total > 0:
        step = 0
    
    for ticker_symbol, data in ticker_data.items():
        try:
            #Process each symbol one by one then
            #clear keras backend before moving to next letter
            batch_train(ticker_symbol, data)
        except Exception as e:
            print(f"Error training model for {ticker_symbol}: {e}")
        finally:
            if progress_callback and total > 0:
                step += 1
                progress_callback(step / total * 100)
        gc.collect()
    print("All models trained successfully.")

if __name__ == "__main__":
    train_models(pull_data=True)