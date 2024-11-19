import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from datetime import datetime, timedelta 
from time import sleep

def get_data():
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    parent_dir = os.path.dirname(os.path.dirname(__file__))

    ticker_list = None
    with open(os.path.join(parent_dir, 'tickers.txt'), 'r') as f:
        ticker_list = f.read()

    ticker_list = ticker_list.split('\n')
    ticker_list = [symb.strip() for symb in ticker_list]

    all_data = {}
    for ticker in ticker_list:
        try:
            print(f"downloading data for {ticker}")
            data = yf.download(ticker, start='2013-01-01', end=end_date)
            all_data[ticker] = data
            sleep(1)
        except Exception as e:
            print(f"could not get data for {ticker}: {e}")

    combined_data = pd.concat(all_data, axis=1)
    combined_data.to_csv(os.path.join(os.path.dirname(__file__),'all_stock_data.csv'))

if __name__ == "__main__":
    get_data()