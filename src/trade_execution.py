from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os
import json
from cryptography.fernet import Fernet
import pandas as pd

ALPACA_CREDENTIALS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'credentials', 'alpaca_credentials.json')
ENCRYPTION_KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'credentials', 'encryption_key.key')

def load_credentials():
    if not os.path.exists(ALPACA_CREDENTIALS_PATH) or not os.path.exists(ENCRYPTION_KEY_PATH):
        return False
    try:
        # Load the encryption key
        with open(ENCRYPTION_KEY_PATH, "rb") as key_file:
            encryption_key = key_file.read()

        cipher = Fernet(encryption_key)

        # Load and decrypt credentials
        with open(ALPACA_CREDENTIALS_PATH, "rb") as f:
            encrypted_data = f.read()
        decrypted_data = cipher.decrypt(encrypted_data).decode()
        return json.loads(decrypted_data)
    except Exception as e:
        print(f"Error loading credentials - {str(e)}")
        return False

def execute_trades():
    credentials = load_credentials()
    if not credentials:
        print("Failed to load API credentials")
        return

    API_KEY = credentials.get("api_key")
    API_SECRET = credentials.get("api_secret")

    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

    # Validate the credentials
    if not API_KEY or not API_SECRET:
        print("API Key or Secret missing.")
        return

    # Load the buy/sell decisions CSV
    csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'buy_sell_decisions.csv')
    if not os.path.exists(csv_file):
        print("Error: Buy/Sell decisions file not found.")
        return

    decisions = pd.read_csv(csv_file)

    # Loop through the CSV and execute trades
    for index, row in decisions.iterrows():
        ticker = row['ticker']
        action = row['action']
        quantity = 1  # For simplicity, assume 1 share per trade (can be parameterized)

        try:
            if action == "Buy":
                market_order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=quantity,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                market_order = trading_client.submit_order(order_data=market_order_data)
                print(f"Executed Buy for {ticker}")
            elif action == "Sell":
                try:
                    position = trading_client.get_open_position(ticker)
                    owned_qty = position.qty
                    if owned_qty > 0:
                        market_order_data = MarketOrderRequest(
                            symbol=ticker,
                            qty=quantity,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY
                        )
                        market_order = trading_client.submit_order(order_data=market_order_data)
                        print(f"Executed Sell for {ticker}")
                    else:
                        print(f"Skipped Sell for {ticker} (No shares owned)")
                except Exception as e:
                    # Handle cases where the position does not exist
                    print(f"Skipped Sell for {ticker} (No position found): {str(e)}")
            else:
                print(f"Skipped {ticker} (Hold action)")
        except Exception as e:
            print(f"Error executing trade for {ticker}: {str(e)}")

if __name__ == "__main__":
    execute_trades()