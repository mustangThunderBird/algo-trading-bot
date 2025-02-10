import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from model.model_manager import ModelManager

class TestModelManager(unittest.TestCase):
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def setUp(self, mock_read_csv, mock_exists, mock_makedirs):
        # Mock the sentiment file existence and content
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({
            'sentiment_score': [0.1, 0.2, 0.3]
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        self.sentiment_file = 'fake_sentiment_file.csv'
        self.quant_model_dir = 'fake_model_dir'
        self.model_manager = ModelManager(self.sentiment_file, self.quant_model_dir)

    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_model_generator(self, mock_pickle_load, mock_open, mock_listdir):
        # Mock the model files in the directory
        mock_listdir.return_value = ['AAPL_model.pkl', 'GOOGL_model.pkl']
        mock_pickle_load.side_effect = ['model_AAPL', 'model_GOOGL']

        models = list(self.model_manager.model_generator())
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0], ('AAPL', 'model_AAPL'))
        self.assertEqual(models[1], ('GOOGL', 'model_GOOGL'))

    @patch('model.model_handler.predict_ticker')
    @patch('builtins.open', new_callable=mock_open)
    def test_make_decisions(self, mock_open, mock_predict_ticker):
        # Mock the model generator
        self.model_manager.model_generator = MagicMock(return_value=[
            ('AAPL', 'model_AAPL'),
            ('GOOGL', 'model_GOOGL')
        ])
        mock_predict_ticker.side_effect = [0.05, 0.1]

        output_file = 'fake_output_file.csv'
        self.model_manager.make_decisions(output_file)

        mock_open.assert_called_once_with(output_file, 'w')
        handle = mock_open()
        handle.write.assert_any_call("ticker,next_day_return,sentiment_score,decision_score,action\n")
        #handle.write.assert_any_call("AAPL,0.05,0.1,4.35,Buy\n")
        #handle.write.assert_any_call("GOOGL,0.1,0.2,8.7,Buy\n")

if __name__ == '__main__':
    unittest.main()