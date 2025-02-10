import unittest
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'src'))
from model.qualitative.qual_model import determine_sentiments, basic_cleanup, add_sentiment, get_article_text, preprocess_data, preprocess_and_update
import pandas as pd

class TestQualModel(unittest.TestCase):

    @patch('model.qualitative.qual_model.fetch_news')
    @patch('model.qualitative.qual_model.get_article_text')
    @patch('model.qualitative.qual_model.pipeline')
    def test_update_progress(self, mock_pipeline, mock_get_article_text, mock_fetch_news):
        # Mock progress callback
        progress_callback = Mock()

        # Mock fetch_news to return a controlled set of data
        mock_fetch_news.return_value = {
            'AAPL': pd.DataFrame({'link': ['http://example.com'], 'summary': ['summary'], 'published': ['date'], 'title': ['title']}),
            'GOOGL': pd.DataFrame({'link': ['http://example.com'], 'summary': ['summary'], 'published': ['date'], 'title': ['title']})
        }

        # Mock get_article_text to return a controlled text
        mock_get_article_text.return_value = "This is a test article text."

        # Mock sentiment_pipeline to return a controlled sentiment
        mock_pipeline.return_value = lambda text: [{'label': 'POSITIVE'}]

        # Call determine_sentiments with the mock progress callback
        determine_sentiments(progress_callback=progress_callback)

        # Check if progress callback was called with the correct progress percentage
        progress_callback.assert_called_with(100.0)

if __name__ == '__main__':
    unittest.main()