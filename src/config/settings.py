"""General settings for the Bitcoin Festival Price Tracker."""

from pathlib import Path

# Data settings
DATA_DIR = Path(__file__).parent.parent.parent / 'data'
DATABASE_PATH = DATA_DIR / 'bitcoin_festivals.db'

# Bitcoin settings
BITCOIN_TICKER = 'BTC-USD'
START_DATE = '2014-01-01'

# API settings
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Chart settings
CHART_COLORS = {
    'price': '#1f77b4',
    'positive': 'green',
    'negative': 'red',
    'neutral': 'yellow'
} 