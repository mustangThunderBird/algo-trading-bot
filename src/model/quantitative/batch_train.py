import argparse
import matplotlib
import os
import warnings
import quant_model as qm
from tensorflow.keras import backend as K
from memory_profiler import profile
import gc

@profile
def batch_train(t, d):
    results = qm.build_quant_model(t, d)
    if results:
        print(f"Successfully built model for {t}")
    K.clear_session()
    del results
    

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train models with optional data pull.")
    parser.add_argument('--pull', action='store_true', help="Pull data using data_download.py")
    args = parser.parse_args()

    # Handle the --pull flag
    if args.pull:
        from data_download import get_data
        print("Pulling data using data_download.py...")
        get_data()

    warnings.filterwarnings('ignore')
    os.system("export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")
    matplotlib.use('Agg')
    filepath = os.path.join(os.path.dirname(__file__), 'all_stock_data.csv')
    ticker_data = qm.preprocess_all_stocks_data(filepath=filepath)
    
    for ticker_symbol, data in ticker_data.items():
        #Process each symbol one by one then
        #clear keras backend before moving to next letter
        batch_train(ticker_symbol, data)
        gc.collect()
    
