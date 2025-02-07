import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'src'))
from model.quantitative.batch_train import train_models

class TestBatchTrain(unittest.TestCase):

    @patch('model.quantitative.batch_train.qm.preprocess_all_stocks_data')
    @patch('model.quantitative.batch_train.batch_train')
    @patch('model.quantitative.data_download.get_data')
    def test_train_models(self, mock_get_data, mock_batch_train, mock_preprocess_all_stocks_data):
        # Mock the data returned by preprocess_all_stocks_data
        mock_preprocess_all_stocks_data.return_value = {
            'AAPL': 'mock_data_1',
            'GOOGL': 'mock_data_2'
        }

        # Mock the progress callback
        progress_callback = MagicMock()

        # Call the function with pull_data=True
        train_models(pull_data=True, progress_callback=progress_callback)

        # Check if get_data was called
        mock_get_data.assert_called_once()

        # Check if preprocess_all_stocks_data was called
        mock_preprocess_all_stocks_data.assert_called_once()

        # Check if batch_train was called for each ticker
        self.assertEqual(mock_batch_train.call_count, 2)
        mock_batch_train.assert_any_call('AAPL', 'mock_data_1')
        mock_batch_train.assert_any_call('GOOGL', 'mock_data_2')

        # Check if progress_callback was called
        self.assertEqual(progress_callback.call_count, 2)

if __name__ == '__main__':
    unittest.main()