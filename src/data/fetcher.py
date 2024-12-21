"""Data fetcher module for retrieving Bitcoin prices and festival data."""

import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List

from config.settings import BITCOIN_TICKER, START_DATE, MAX_RETRIES, RETRY_DELAY
from config.festivals import FESTIVALS

def fetch_bitcoin_prices(start_date: str = START_DATE, end_date: str = None) -> pd.DataFrame:
    """Fetch Bitcoin price data from Yahoo Finance.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today)
        
    Returns:
        DataFrame with Bitcoin prices
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    for attempt in range(MAX_RETRIES):
        try:
            btc = yf.Ticker(BITCOIN_TICKER)
            df = btc.history(start=start_date, end=end_date)
            if not df.empty:
                # Convert to timezone-naive
                df.index = df.index.tz_localize(None)
                return df[['Close']].rename(columns={'Close': 'price'})
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise Exception(f"Failed to fetch Bitcoin prices: {str(e)}")
            time.sleep(RETRY_DELAY)
    
    return pd.DataFrame()

def get_festivals_data() -> pd.DataFrame:
    """Get festival data as a DataFrame.
    
    Returns:
        DataFrame containing festival information
    """
    festivals_data = []
    for category, festivals in FESTIVALS.items():
        for festival in festivals:
            festivals_data.append({
                'name': festival['name'],
                'category': category,
                'start_date': festival['start_date'],
                'end_date': festival['end_date']
            })
    return pd.DataFrame(festivals_data)

def analyze_festival_performance(prices_df: pd.DataFrame, festival_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze Bitcoin price performance during festivals.
    
    Args:
        prices_df: DataFrame with Bitcoin prices
        festival_df: DataFrame with festival dates
        
    Returns:
        DataFrame with festival performance metrics
    """
    # Ensure price index is timezone naive
    if prices_df.index.tz is not None:
        prices_df.index = prices_df.index.tz_localize(None)
    
    stats = []
    for _, festival in festival_df.iterrows():
        # Convert dates to timezone-naive if they aren't already
        start_date = pd.to_datetime(festival['start_date']).tz_localize(None).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(festival['end_date']).tz_localize(None).strftime('%Y-%m-%d')
        
        # Filter prices using string dates
        festival_prices = prices_df[start_date:end_date]
        if not festival_prices.empty:
            min_idx = festival_prices['price'].idxmin()
            max_idx = festival_prices['price'].idxmax()
            
            stats.append({
                'Festival': festival['name'],
                'Category': festival['category'],
                'Start Date': start_date,
                'End Date': end_date,
                'Start Price': festival_prices['price'].iloc[0],
                'End Price': festival_prices['price'].iloc[-1],
                'Min Price': festival_prices['price'].min(),
                'Max Price': festival_prices['price'].max(),
                'Best Buy Date': min_idx,
                'Best Sell Date': max_idx,
                'Price Change %': ((festival_prices['price'].iloc[-1] - festival_prices['price'].iloc[0]) 
                                 / festival_prices['price'].iloc[0] * 100),
                'Max Return %': ((festival_prices['price'].max() - festival_prices['price'].min()) 
                               / festival_prices['price'].min() * 100),
                'Volatility': festival_prices['price'].std() / festival_prices['price'].mean() * 100
            })
    
    return pd.DataFrame(stats) 