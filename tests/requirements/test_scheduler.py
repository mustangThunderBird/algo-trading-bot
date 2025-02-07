import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'src'))
from scheduler import Scheduler

class TestScheduler(unittest.TestCase):

    @patch('scheduler.qual_model.determine_sentiments')
    @patch('scheduler.logging')
    def test_run_qualitative_model_success(self, mock_logging, mock_determine_sentiments):
        scheduler = Scheduler()
        scheduler.run_qualitative_model()
        
        mock_logging.info.assert_any_call("Starting qualitative model execution...")
        mock_determine_sentiments.assert_called_once()
        mock_logging.info.assert_any_call("Qualitative model execution completed successfully")

    @patch('scheduler.qual_model.determine_sentiments', side_effect=Exception("Test exception"))
    @patch('scheduler.logging')
    def test_run_qualitative_model_failure(self, mock_logging, mock_determine_sentiments):
        scheduler = Scheduler()
        scheduler.run_qualitative_model()
        
        mock_logging.info.assert_any_call("Starting qualitative model execution...")
        mock_determine_sentiments.assert_called_once()
        mock_logging.error.assert_called_once_with("Error running qualitative model: Test exception")

if __name__ == '__main__':
    unittest.main()