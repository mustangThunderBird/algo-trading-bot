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
    "Sign in to access your portfolio"]

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
        futures = {executor.submit(fetch_news, t): t for t in ticker_list}
        
        for future in as_completed(futures):
            ticker, df = future.result()
            sn_frames[ticker] = df

def basic_cleanup(text):
    '''Perform basic cleanup on article text'''
    cleaned_text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8', 'ignore')
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"()\s]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def summarize_article(article_text):
    summary = summarizer(article_text[:2056], max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def add_sentiment(df:pd.DataFrame):
    setiments = df['article_text'].apply(lambda text: sentiment_pipeline(text[:2056])[0]['label'] if text != "N/A" else "UNKNOWN")
    df['sentiment'] = setiments
    return df

def get_article_text(url):
    try:     
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragprahs = soup.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragprahs])
            article_text = basic_cleanup(article_text)
            return article_text
        else:
            print(f"Failed to get article text from {url} got a response code of {response.status_code}")
            return "N/A"
    except Exception as e:
        print(f"Failed to get article text from {url}: {e}")
        return "N/A"

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    try:
        df = df[['summary', 'link', 'published', 'title']]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            article_texts = list(executor.map(get_article_text, df['link']))
        df['article_text'] = article_texts
        df = add_sentiment(df)
        return df
    except Exception as e:
        return f"Could not process data {e}"