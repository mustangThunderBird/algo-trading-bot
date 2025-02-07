import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from trade_execution import load_credentials, execute_trades

class TestTradeExecution(unittest.TestCase):

    @patch('trade_execution.os.path.exists')
    @patch('trade_execution.open')
    @patch('trade_execution.Fernet')
    def test_load_credentials_success(self, mock_fernet, mock_open, mock_exists):
        mock_exists.side_effect = [True, True]
        mock_key_file = MagicMock()
        mock_open.side_effect = [mock_key_file, mock_key_file]
        mock_key_file.read.return_value = b'some_encryption_key'
        mock_cipher = MagicMock()
        mock_fernet.return_value = mock_cipher
        mock_cipher.decrypt.return_value = b'{"api_key": "test_key", "api_secret": "test_secret"}'

        credentials = load_credentials()
        self.assertEqual(credentials, {"api_key": "test_key", "api_secret": "test_secret"})

    @patch('trade_execution.os.path.exists')
    def test_load_credentials_missing_files(self, mock_exists):
        mock_exists.side_effect = [False, False]
        credentials = load_credentials()
        self.assertFalse(credentials)

    @patch('trade_execution.os.path.exists')
    @patch('trade_execution.open')
    @patch('trade_execution.Fernet')
    def test_load_credentials_decryption_error(self, mock_fernet, mock_open, mock_exists):
        mock_exists.side_effect = [True, True]
        mock_key_file = MagicMock()
        mock_open.side_effect = [mock_key_file, mock_key_file]
        mock_key_file.read.return_value = b'some_encryption_key'
        mock_cipher = MagicMock()
        mock_fernet.return_value = mock_cipher
        mock_cipher.decrypt.side_effect = Exception("Decryption error")

        credentials = load_credentials()
        self.assertFalse(credentials)

    @patch('trade_execution.load_credentials')
    @patch('trade_execution.TradingClient')
    @patch('trade_execution.pd.read_csv')
    @patch('trade_execution.os.path.exists')
    def test_execute_trades_success(self, mock_exists, mock_read_csv, mock_trading_client, mock_load_credentials):
        mock_load_credentials.return_value = {"api_key": "test_key", "api_secret": "test_secret"}
        mock_exists.side_effect = [True, True]
        mock_read_csv.return_value = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'action': ['Buy', 'Sell']
        })
        mock_client_instance = MagicMock()
        mock_trading_client.return_value = mock_client_instance

        result = execute_trades()
        self.assertEqual(result, 0)
        self.assertEqual(mock_client_instance.submit_order.call_count, 1)

    @patch('trade_execution.load_credentials')
    @patch('trade_execution.os.path.exists')
    def test_execute_trades_missing_credentials(self, mock_exists, mock_load_credentials):
        mock_load_credentials.return_value = False
        result = execute_trades()
        self.assertEqual(result, -1)

    @patch('trade_execution.load_credentials')
    @patch('trade_execution.os.path.exists')
    def test_execute_trades_missing_csv(self, mock_exists, mock_load_credentials):
        mock_load_credentials.return_value = {"api_key": "test_key", "api_secret": "test_secret"}
        mock_exists.side_effect = [False, False]
        result = execute_trades()
        self.assertEqual(result, -3)

if __name__ == '__main__':
    unittest.main()