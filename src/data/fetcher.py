"""Data fetching and analysis module."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import requests
import time
import os
from config.festivals import FESTIVALS
from config.events import HALVING_EVENTS, MAJOR_EVENTS
from data.database import Database

# Constants
START_DATE = "2014-01-01"
BITCOIN_SYMBOL = "BTC-USD"
BITCOIN_TICKER = "BTC-USD"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

class DataFetcher:
    """Class for fetching and analyzing Bitcoin price data."""
    
    def __init__(self, db: Optional[Database] = None):
        """Initialize the data fetcher.
        
        Args:
            db: Optional Database instance. If not provided, a new one will be created.
        """
        self.db = db if db is not None else Database()
        
    def fetch_bitcoin_prices(self, start_date: str = START_DATE, end_date: str = None) -> pd.DataFrame:
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

    def get_festivals_data(self) -> pd.DataFrame:
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

    def calculate_max_drawdown(self, prices: pd.Series) -> tuple:
        """Calculate the maximum drawdown and its duration.
        
        Args:
            prices: Series of prices
            
        Returns:
            tuple: (max_drawdown_pct, duration_days, peak_date, trough_date)
        """
        # Calculate cumulative max
        rolling_max = prices.expanding().max()
        drawdowns = prices / rolling_max - 1
        
        # Find the maximum drawdown
        max_drawdown = drawdowns.min()
        
        # Find the peak and trough dates
        peak_idx = prices[:drawdowns.idxmin()].idxmax()
        trough_idx = drawdowns.idxmin()
        
        # Calculate duration
        duration_days = (trough_idx - peak_idx).days
        
        return (
            max_drawdown * 100,  # Convert to percentage
            duration_days,
            peak_idx,
            trough_idx
        )

    def calculate_weekly_volatility(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weekly volatility of Bitcoin prices.
        
        Args:
            prices_df: DataFrame with Bitcoin prices
            
        Returns:
            DataFrame with weekly volatility metrics
        """
        # Resample to weekly frequency
        weekly_data = prices_df.resample('W').agg({
            'price': ['first', 'last', 'min', 'max', 'std', 'mean']
        })
        
        weekly_data.columns = ['Open', 'Close', 'Low', 'High', 'Std', 'Mean']
        
        # Calculate weekly metrics
        weekly_data['Volatility'] = (weekly_data['Std'] / weekly_data['Mean']) * 100
        weekly_data['Range %'] = ((weekly_data['High'] - weekly_data['Low']) / weekly_data['Low']) * 100
        weekly_data['Return %'] = ((weekly_data['Close'] - weekly_data['Open']) / weekly_data['Open']) * 100
        
        return weekly_data

    def analyze_festival_performance(self, prices_df: pd.DataFrame, festivals_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze Bitcoin price performance during festivals.
        
        Args:
            prices_df: DataFrame with Bitcoin prices
            festivals_df: DataFrame with festival dates
            
        Returns:
            DataFrame with festival performance statistics
        """
        # Ensure price index is timezone naive
        if prices_df.index.tz is not None:
            prices_df.index = prices_df.index.tz_localize(None)
        
        stats = []
        for _, festival in festivals_df.iterrows():
            # Convert dates to timezone-naive if they aren't already
            start_date = pd.to_datetime(festival['start_date']).tz_localize(None)
            end_date = pd.to_datetime(festival['end_date']).tz_localize(None)
            
            # Calculate festival duration
            duration = (end_date - start_date).days + 1  # Include both start and end dates
            
            # Filter prices using string dates
            festival_prices = prices_df[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]
            if not festival_prices.empty:
                min_idx = festival_prices['price'].idxmin()
                max_idx = festival_prices['price'].idxmax()
                
                stats.append({
                    'Festival': festival['name'],
                    'Category': festival['category'],
                    'Start Date': start_date.strftime('%Y-%m-%d'),
                    'End Date': end_date.strftime('%Y-%m-%d'),
                    'Duration': duration,
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
        
        stats_df = pd.DataFrame(stats)
        
        # Add max drawdown analysis
        stats_df['Max Drawdown %'] = float('nan')
        stats_df['Drawdown Duration'] = float('nan')
        stats_df['Drawdown Peak'] = pd.NaT
        stats_df['Drawdown Trough'] = pd.NaT
        
        for idx, row in stats_df.iterrows():
            festival_prices = prices_df[
                (prices_df.index >= pd.to_datetime(row['Start Date'])) &
                (prices_df.index <= pd.to_datetime(row['End Date']))
            ]['price']
            
            if not festival_prices.empty:
                drawdown, duration, peak, trough = self.calculate_max_drawdown(festival_prices)
                stats_df.at[idx, 'Max Drawdown %'] = drawdown
                stats_df.at[idx, 'Drawdown Duration'] = duration
                stats_df.at[idx, 'Drawdown Peak'] = peak
                stats_df.at[idx, 'Drawdown Trough'] = trough
        
        # Add weekly volatility analysis
        stats_df['Avg Weekly Volatility'] = float('nan')
        stats_df['Max Weekly Volatility'] = float('nan')
        stats_df['Max Weekly Range %'] = float('nan')
        
        for idx, row in stats_df.iterrows():
            festival_prices = prices_df[
                (prices_df.index >= pd.to_datetime(row['Start Date'])) &
                (prices_df.index <= pd.to_datetime(row['End Date']))
            ]
            
            if not festival_prices.empty:
                weekly_stats = self.calculate_weekly_volatility(festival_prices)
                stats_df.at[idx, 'Avg Weekly Volatility'] = weekly_stats['Volatility'].mean()
                stats_df.at[idx, 'Max Weekly Volatility'] = weekly_stats['Volatility'].max()
                stats_df.at[idx, 'Max Weekly Range %'] = weekly_stats['Range %'].max()
        
        return stats_df 

    def identify_market_cycles(self, prices_df: pd.DataFrame, threshold: float = 20.0) -> pd.DataFrame:
        """
        Identify bull and bear market cycles based on price movements.
        A bull market starts when price increases by threshold% from a low
        A bear market starts when price decreases by threshold% from a high
        """
        cycles = []
        current_type = None
        cycle_start = prices_df.index[0]
        cycle_start_price = prices_df['price'].iloc[0]
        last_extreme = cycle_start_price
        
        for date, row in prices_df.iterrows():
            price = row['price']
            
            if current_type is None:
                # Initialize first cycle type based on first price movement
                if price > cycle_start_price * (1 + threshold/100):
                    current_type = 'Bull'
                elif price < cycle_start_price * (1 - threshold/100):
                    current_type = 'Bear'
            else:
                if current_type == 'Bull' and price < last_extreme * (1 - threshold/100):
                    # Bull market ends, bear market begins
                    cycles.append({
                        'Type': 'Bull',
                        'Start Date': cycle_start,
                        'End Date': date,
                        'Start Price': cycle_start_price,
                        'End Price': price,
                        'Return %': (price - cycle_start_price) / cycle_start_price * 100,
                        'Duration': (date - cycle_start).days
                    })
                    cycle_start = date
                    cycle_start_price = price
                    current_type = 'Bear'
                    last_extreme = price
                elif current_type == 'Bear' and price > last_extreme * (1 + threshold/100):
                    # Bear market ends, bull market begins
                    cycles.append({
                        'Type': 'Bear',
                        'Start Date': cycle_start,
                        'End Date': date,
                        'Start Price': cycle_start_price,
                        'End Price': price,
                        'Return %': (price - cycle_start_price) / cycle_start_price * 100,
                        'Duration': (date - cycle_start).days
                    })
                    cycle_start = date
                    cycle_start_price = price
                    current_type = 'Bull'
                    last_extreme = price
            
            # Update last extreme price
            if current_type == 'Bull':
                last_extreme = max(last_extreme, price)
            elif current_type == 'Bear':
                last_extreme = min(last_extreme, price)
        
        # Add final cycle
        if current_type:
            cycles.append({
                'Type': current_type,
                'Start Date': cycle_start,
                'End Date': prices_df.index[-1],
                'Start Price': cycle_start_price,
                'End Price': prices_df['price'].iloc[-1],
                'Return %': (prices_df['price'].iloc[-1] - cycle_start_price) / cycle_start_price * 100,
                'Duration': (prices_df.index[-1] - cycle_start).days
            })
        
        return pd.DataFrame(cycles)

    def get_halving_dates(self) -> List[pd.Timestamp]:
        """Return list of Bitcoin halving dates."""
        from config.events import HALVING_EVENTS
        
        # Convert datetime objects to pandas Timestamps
        halving_dates = [pd.Timestamp(event['date']) for event in HALVING_EVENTS]
        return sorted(halving_dates)

    def get_next_halving_date(self) -> pd.Timestamp:
        """Get the next Bitcoin halving date."""
        now = pd.Timestamp.now()
        halving_dates = self.get_halving_dates()
        
        # Find the next halving date after today
        for date in halving_dates:
            if date > now:
                return date
            
        return None

    def analyze_halving_impact(self, prices_df: pd.DataFrame, window_days: int = 180) -> pd.DataFrame:
        """
        Analyze Bitcoin price behavior around halving events.
        Returns DataFrame with pre and post halving statistics.
        """
        halving_stats = []
        now = pd.Timestamp.now()
        
        # Get the date range from the prices DataFrame
        start_date = prices_df.index[0]
        end_date = prices_df.index[-1]
        
        for halving_date in self.get_halving_dates():
            # Skip if halving date is outside our analysis range
            if halving_date < start_date - pd.Timedelta(days=window_days) or halving_date > end_date + pd.Timedelta(days=window_days):
                continue
            
            # For past halvings
            if halving_date <= now:
                # Get price at halving - find the closest available price
                halving_prices = prices_df[prices_df.index <= halving_date]
                if not halving_prices.empty:
                    halving_price = halving_prices['price'].iloc[-1]
                else:
                    halving_price = None
                
                # Pre-halving analysis
                pre_halving = prices_df[
                    (prices_df.index >= halving_date - pd.Timedelta(days=window_days)) &
                    (prices_df.index <= halving_date)
                ]
                
                if not pre_halving.empty:
                    pre_return = (halving_price - pre_halving['price'].iloc[0]) / pre_halving['price'].iloc[0] * 100
                    pre_volatility = pre_halving['price'].pct_change().std() * np.sqrt(252) * 100
                else:
                    pre_return = None
                    pre_volatility = None
                
                # Post-halving analysis
                post_halving = prices_df[
                    (prices_df.index >= halving_date) &
                    (prices_df.index <= halving_date + pd.Timedelta(days=window_days))
                ]
                
                if not post_halving.empty and halving_price is not None:
                    post_return = (post_halving['price'].iloc[-1] - halving_price) / halving_price * 100
                    post_volatility = post_halving['price'].pct_change().std() * np.sqrt(252) * 100
                    
                    # Get price after 1 year if available
                    one_year_later = prices_df[
                        (prices_df.index >= halving_date) &
                        (prices_df.index <= halving_date + pd.Timedelta(days=365))
                    ]
                    price_after_1yr = one_year_later['price'].iloc[-1] if len(one_year_later) > 0 else None
                else:
                    post_return = None
                    post_volatility = None
                    price_after_1yr = None
                
            # For future halvings
            else:
                # Calculate pre-halving metrics using available data
                pre_halving_start = halving_date - pd.Timedelta(days=window_days)
                available_data = prices_df[prices_df.index >= pre_halving_start]
                
                if not available_data.empty:
                    pre_return = ((available_data['price'].iloc[-1] - available_data['price'].iloc[0]) 
                                / available_data['price'].iloc[0] * 100)
                    pre_volatility = available_data['price'].pct_change().std() * np.sqrt(252) * 100
                else:
                    pre_return = None
                    pre_volatility = None
                
                halving_price = None
                post_return = None
                post_volatility = None
                price_after_1yr = None
            
            # Only add to stats if we have some data
            if pre_return is not None or post_return is not None:
                halving_stats.append({
                    'Halving Date': halving_date,
                    'Pre-Halving Return %': pre_return,
                    'Post-Halving Return %': post_return,
                    'Pre-Halving Volatility': pre_volatility,
                    'Post-Halving Volatility': post_volatility,
                    'Price at Halving': halving_price,
                    'Price After 1 Year': price_after_1yr,
                    'Is Future': halving_date > now
                })
        
        return pd.DataFrame(halving_stats)

    def get_major_events(self) -> List[Dict]:
        """Return list of major market events that impacted Bitcoin price."""
        from config.events import MAJOR_EVENTS
        
        # Convert datetime objects to pandas Timestamps
        events = []
        for event in MAJOR_EVENTS:
            events.append({
                **event,
                'date': pd.Timestamp(event['date'])
            })
        
        return sorted(events, key=lambda x: x['date'])

    def analyze_event_impact(self, prices_df: pd.DataFrame, event: Dict, window_days: int = 30) -> Dict:
        """Analyze price impact of a major market event.
        
        Args:
            prices_df: DataFrame with Bitcoin prices
            event: Dictionary containing event information
            window_days: Number of days to analyze before and after event
            
        Returns:
            Dictionary with impact analysis
        """
        event_date = event['date']
        
        # Get prices around the event
        pre_event = prices_df[
            (prices_df.index >= event_date - pd.Timedelta(days=window_days)) &
            (prices_df.index <= event_date)
        ]
        
        post_event = prices_df[
            (prices_df.index >= event_date) &
            (prices_df.index <= event_date + pd.Timedelta(days=window_days))
        ]
        
        if pre_event.empty or post_event.empty:
            return None
        
        # Calculate impact metrics
        event_price = prices_df[prices_df.index <= event_date]['price'].iloc[-1]
        pre_price = pre_event['price'].iloc[0]
        post_price = post_event['price'].iloc[-1]
        
        return {
            **event,
            'price_at_event': event_price,
            'pre_event_return': ((event_price - pre_price) / pre_price) * 100,
            'post_event_return': ((post_price - event_price) / event_price) * 100,
            'pre_event_volatility': pre_event['price'].pct_change().std() * np.sqrt(252) * 100,
            'post_event_volatility': post_event['price'].pct_change().std() * np.sqrt(252) * 100
        }

    def fetch_fear_greed_index(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch Fear & Greed Index data from alternative.me API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with Fear & Greed Index data
        """
        try:
            # Calculate number of days to fetch
            if start_date and end_date:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                days = (end - start).days + 1
                limit = max(days, 30)  # Fetch at least 30 days
            else:
                limit = 365  # Default to 1 year
            
            # Construct API URL with dynamic limit
            url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
            
            # Make request with retries
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    break
                except requests.RequestException as e:
                    if attempt == 2:  # Last attempt
                        raise Exception(f"Failed to fetch Fear & Greed Index data: {str(e)}")
                    time.sleep(1)  # Wait before retry
            
            # Parse response
            if data.get('metadata', {}).get('error') is not None:
                raise Exception(f"API Error: {data['metadata']['error']}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Convert value to numeric first
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df.set_index('timestamp', inplace=True)
            df.index = df.index.tz_localize(None)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            # Add value classification
            df['value_classification'] = df['value'].apply(self.get_fear_greed_label)
            
            return df
        
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def get_fear_greed_label(self, value: int) -> str:
        """Get sentiment label for Fear & Greed Index value."""
        if value <= 10:
            return "Maximum Fear"
        elif value <= 25:
            return "Extreme Fear"
        elif value <= 45:
            return "Fear"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        elif value <= 90:
            return "Extreme Greed"
        else:
            return "Maximum Greed"

    def _generate_market_summary(self, valuation: Dict, risk: Dict, cycle: Dict) -> str:
        """Generate a market summary based on various metrics."""
        try:
            mvrv = valuation['mvrv_zscore']
            risk_level = risk['risk_signal']
            phase = cycle['current_phase']
            
            if mvrv > 5:
                sentiment = "extremely overvalued"
            elif mvrv > 2:
                sentiment = "overvalued"
            elif mvrv < -1:
                sentiment = "undervalued"
            elif mvrv < -2:
                sentiment = "extremely undervalued"
            else:
                sentiment = "fairly valued"
            
            return f"The market is currently in a {phase.lower()} phase and appears to be {sentiment}. " \
                   f"Risk levels are {risk_level.lower()}, with the market showing {cycle['cycle_progress']}% " \
                   f"progression through the current cycle. {cycle['cycle_prediction']}."
                   
        except Exception:
            return "Unable to generate market summary due to insufficient data."

    def calculate_drawdown(self, prices: pd.Series) -> pd.Series:
        """Calculate rolling drawdown for a price series.
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of drawdown values in percentage
        """
        # Calculate rolling maximum
        rolling_max = prices.expanding().max()
        
        # Calculate drawdown percentage
        drawdown = (prices / rolling_max - 1) * 100
        
        return drawdown

    def fetch_nupl_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch Net Unrealized Profit/Loss (NUPL) data from bitcoin-data.com API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with NUPL data
        """
        try:
            # Check if we need to update the data
            last_update = self.db.get_last_nupl_update()
            now = pd.Timestamp.now()
            
            # Only update once per day
            if last_update is None or (now - last_update).days >= 1:
                url = "https://bitcoin-data.com/v1/nupl"
                
                # Make request with retries
                for attempt in range(MAX_RETRIES):
                    try:
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        data = response.json()
                        break
                    except requests.RequestException as e:
                        if attempt == MAX_RETRIES - 1:
                            raise Exception(f"Failed to fetch NUPL data: {str(e)}")
                        time.sleep(RETRY_DELAY)
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Convert NUPL from string to float
                df['nupl'] = pd.to_numeric(df['nupl'], errors='coerce')
                
                # Convert date to datetime
                df['date'] = pd.to_datetime(df['d'])
                
                # Drop unnecessary columns and set index
                df = df.drop(['d', 'unixTs'], axis=1)
                df.set_index('date', inplace=True)
                
                # Sort by date
                df.sort_index(inplace=True)
                
                # Store in database
                self.db.store_nupl_data(df)
            
            # Get data from database
            df = self.db.get_nupl_data()  # Get all data first
            
            if df.empty:
                return df
            
            # Filter by date range if provided
            if start_date:
                start_ts = pd.to_datetime(start_date)
                df = df[df.index >= start_ts]
            if end_date:
                end_ts = pd.to_datetime(end_date)
                df = df[df.index <= end_ts]
            
            # If no data in range, return empty DataFrame
            if df.empty:
                return df
            
            # Add NUPL classification
            df['nupl_classification'] = df['nupl'].apply(self.get_nupl_label)
            
            return df
        
        except Exception as e:
            print(f"Error fetching NUPL data: {str(e)}")
            return pd.DataFrame()

    def get_nupl_label(self, value: float) -> str:
        """Get market phase label based on NUPL value.
        
        Args:
            value: NUPL value
            
        Returns:
            Market phase label
        """
        if value <= -0.25:
            return "Capitulation"
        elif value <= 0:
            return "Fear & Anxiety"
        elif value <= 0.25:
            return "Hope & Optimism"
        elif value <= 0.5:
            return "Belief & Thrill"
        elif value <= 0.75:
            return "Euphoria & Greed"
        else:
            return "Maximum Greed"