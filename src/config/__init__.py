"""Configuration package for Bitcoin Festival Price Tracker."""

from .settings import (
    DATA_DIR,
    DATABASE_PATH,
    BITCOIN_TICKER,
    START_DATE,
    MAX_RETRIES,
    RETRY_DELAY,
    CHART_COLORS
)

from .festivals import (
    FESTIVALS,
    generate_festival_dates
)

__all__ = [
    'DATA_DIR',
    'DATABASE_PATH',
    'BITCOIN_TICKER',
    'START_DATE',
    'MAX_RETRIES',
    'RETRY_DELAY',
    'CHART_COLORS',
    'FESTIVALS',
    'generate_festival_dates'
] 