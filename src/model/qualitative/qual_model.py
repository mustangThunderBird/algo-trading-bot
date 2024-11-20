import requests
from bs4 import BeautifulSoup
from yahoo_fin import news
from transformers import pipeline
import warnings
import pandas as pd
import concurrent.futures
import unicodedata
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from random import uniform

#Initalize transformer pipelines
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=0)
summarizer = pipeline("summarization", device=0)
warnings.filterwarnings('ignore')

filler_texts = [
    "We are experiencing some temporary issues." ,
    "The market data on this page is currently delayed.",
    "Please bear with us as we address this and restore your personalized lists.",
    "Founded in 1993, The Motley Fool is a financial services company dedicated to making the world smarter, happier, and richer.",
    "The Motley Fool reaches millions of people every month through our premium investing solutions, free guidance and market analysis on Fool.com, toprated podcasts, and nonprofit The Motley Fool Foundation.",
    "Founded in 1993, The Motley Fool is a financial services company dedicated to making the world smarter, happier, and richer. The Motley Fool reaches millions of people every month through our premium investing solutions, free guidance and market analysis on Fool.com, toprated podcasts, and nonprofit The Motley Fool Foundation.",
    "You're reading a free article with opinions that may differ from The Motley Fool's Premium Investing Services.",
    "Become a Motley Fool member today to get instant access to our top analyst recommendations, indepth research, investing resources, and more.",
    "Learn More Key Points",
    "Read the latest financial and business news from Yahoo Finance Sign in to access your portfolio",
    "The Motley Fool has a disclosure policy.",
    "Related Articles Invest better with The Motley Fool.",
    "Get stock recommendations, portfolio guidance, and more from The Motley Fool's premium services.",
    "Making the world smarter, happier, and richer.",
    "1995 2024 The Motley Fool. All rights reserved. Market data powered by Xignite and Polygon.io.",
    "About The Motley Fool Our Services Around the Globe Free Tools Affiliates Friends",
    "Free Stock Analysis Report To read this article on Zacks.com click here.",
    "Zacks Investment Research Sign in to access your portfolio",
    "To watch more expert insights and analysis on the latest market action, check out more Catalysts here.",
    "This post was written", 
    "Sign in to access your portfolio"
]

def news_fetcher_helper(ticker):
    stock_news = news.get_yf_rss(ticker)
    return ticker, pd.DataFrame(stock_news)

def fetch_news():
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    ticker_list = None

    with open(os.path.join(parent_dir, 'tickers.txt'), 'r') as f:
        ticker_list = f.read()

    ticker_list = ticker_list.split('\n')
    ticker_list = [symb.strip() for symb in ticker_list]
    sn_frames = {}

    # Run the news fetching in parallel
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(news_fetcher_helper, t): t for t in ticker_list}
        for future in as_completed(futures):
            ticker, df = future.result()
            sn_frames[ticker] = df
    return sn_frames

if __name__ == "__main__":
    stock_news_frames = fetch_news()
    print(stock_news_frames)