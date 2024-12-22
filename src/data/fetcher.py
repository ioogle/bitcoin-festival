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
from config.api_keys import GLASSNODE_API_KEY, check_api_keys

# Constants
START_DATE = "2014-01-01"
BITCOIN_SYMBOL = "BTC-USD"
BITCOIN_TICKER = "BTC-USD"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

class DataFetcher:
    """Class for fetching and analyzing Bitcoin price data."""
    
    def __init__(self):
        """Initialize the data fetcher."""
        # Check if all required API keys are present
        check_api_keys()
        
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

    def fetch_onchain_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch on-chain metrics from various blockchain data APIs.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with on-chain metrics
        """
        try:
            # Convert dates to timestamps
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
            
            # Initialize empty DataFrame with date range
            dates = pd.date_range(start=start_ts, end=end_ts, freq='D')
            data = pd.DataFrame(index=dates)
            
            # Set up Glassnode API
            base_url = "https://api.glassnode.com/v1/metrics"
            headers = {'X-Api-Key': GLASSNODE_API_KEY}
            
            # Define metrics to fetch
            metrics = {
                # Network metrics
                'addresses/active_count': 'active_addresses',
                'transactions/count': 'transaction_count',
                'mining/hash_rate_mean': 'hash_rate',
                'mining/difficulty_latest': 'difficulty',
                'addresses/new_non_zero_count': 'network_growth',
                'transactions/transfers_volume_mean': 'avg_transaction_value',
                'fees/mean': 'avg_transaction_fee',
                'mempool/size': 'mempool_size',
                
                # HODL metrics
                'supply/active_more_1y_percent': 'lth_supply',
                'supply/active_less_1y_percent': 'sth_supply',
                'indicators/coin_days_destroyed': 'coin_days_destroyed',
                'indicators/liveliness': 'liveliness',
                'indicators/average_dormancy': 'dormancy',
                'supply/active_24h': 'supply_last_active_24h',
                
                # Mining metrics
                'mining/revenue_sum': 'miner_revenue',
                'distribution/balance_miners': 'miner_balance',
                'mining/thermocap': 'thermocap',
                'mining/marketcap_thermocap_ratio': 'thermocap_ratio',
                
                # Exchange metrics
                'transactions/transfers_volume_to_exchanges_sum': 'exchange_inflow',
                'transactions/transfers_volume_from_exchanges_sum': 'exchange_outflow',
                'distribution/balance_exchanges': 'exchange_balance',
                'indicators/exchange_supply': 'exchange_supply',
                'derivatives/futures_liquidated_volume_long_sum': 'futures_long_liquidations',
                'derivatives/futures_liquidated_volume_short_sum': 'futures_short_liquidations',
                
                # Market metrics
                'market/marketcap_usd': 'market_cap',
                'market/realized_cap_usd': 'realized_cap',
                'market/mvrv': 'mvrv',
                'market/nvt': 'nvt_ratio',
                'indicators/sopr': 'sopr',
                'indicators/reserve_risk': 'reserve_risk',
                'indicators/puell_multiple': 'puell_multiple'
            }
            
            # Fetch data for each metric with retries
            for endpoint, column in metrics.items():
                for attempt in range(MAX_RETRIES):
                    try:
                        params = {
                            'a': 'BTC',
                            's': int(start_ts.timestamp()),
                            'u': int(end_ts.timestamp()),
                            'i': '24h'  # Daily resolution
                        }
                        
                        response = requests.get(f"{base_url}/{endpoint}", headers=headers, params=params)
                        if response.status_code == 200:
                            metric_data = pd.DataFrame(response.json())
                            if not metric_data.empty:
                                metric_data['t'] = pd.to_datetime(metric_data['t'], unit='s')
                                metric_data.set_index('t', inplace=True)
                                data[column] = metric_data['v']
                            break
                        elif response.status_code == 429:  # Rate limit
                            if attempt < MAX_RETRIES - 1:
                                time.sleep(RETRY_DELAY * (attempt + 1))
                                continue
                        else:
                            print(f"Failed to fetch {column}: {response.status_code}")
                            break
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            print(f"Error fetching {column}: {str(e)}")
                        else:
                            time.sleep(RETRY_DELAY * (attempt + 1))
            
            # Calculate derived metrics
            if 'hash_rate' in data.columns:
                data['hash_ribbons'] = data['hash_rate'].rolling(30).mean() / data['hash_rate'].rolling(60).mean()
            
            if 'difficulty' in data.columns:
                data['difficulty_ribbon'] = data['difficulty'].rolling(30).mean() / data['difficulty'].rolling(60).mean()
            
            if 'miner_revenue' in data.columns and 'hash_rate' in data.columns:
                data['revenue_per_hash'] = data['miner_revenue'] / data['hash_rate']
            
            if 'exchange_inflow' in data.columns and 'exchange_outflow' in data.columns:
                data['net_exchange_flow'] = data['exchange_inflow'] - data['exchange_outflow']
            
            # Fill missing values with forward fill then backward fill
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Convert units for better readability
            unit_conversions = {
                'market_cap': 1e-9,  # Convert to billions
                'realized_cap': 1e-9,
                'miner_revenue': 1e-6,  # Convert to millions
                'thermocap': 1e-9,
                'hash_rate': 1e-6,  # Convert to millions TH/s
                'exchange_inflow': 1e-3,  # Convert to thousands
                'exchange_outflow': 1e-3,
                'exchange_balance': 1e-3,
                'futures_long_liquidations': 1e-6,
                'futures_short_liquidations': 1e-6
            }
            
            for col, factor in unit_conversions.items():
                if col in data.columns:
                    data[col] = data[col] * factor
            
            return data
            
        except Exception as e:
            print(f"Error fetching on-chain data: {str(e)}")
            return pd.DataFrame()

    def analyze_onchain_metrics(self, prices_df: pd.DataFrame, onchain_df: pd.DataFrame) -> Dict:
        """Analyze on-chain metrics and their relationships with price.
        
        Args:
            prices_df: DataFrame with price data
            onchain_df: DataFrame with on-chain metrics
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Calculate current metrics and changes
            current_metrics = {}
            for col in onchain_df.columns:
                current = onchain_df[col].iloc[-1]
                prev = onchain_df[col].iloc[-30] if len(onchain_df) >= 30 else onchain_df[col].iloc[0]
                change = ((current - prev) / prev * 100) if prev != 0 else 0
                current_metrics[col] = current
                current_metrics[f"{col}_change"] = change
            
            # Network metrics analysis
            network = {
                'dates': onchain_df.index,
                'active_addresses': onchain_df['active_addresses'].iloc[-1],
                'active_addresses_change': current_metrics['active_addresses_change'],
                'transaction_count': onchain_df['transaction_count'].iloc[-1],
                'transaction_count_change': current_metrics['transaction_count_change'],
                'hash_rate': onchain_df['hash_rate'].iloc[-1],
                'hash_rate_change': current_metrics['hash_rate_change'],
                'network_growth': onchain_df['network_growth'].iloc[-1],
                'avg_transaction_value': onchain_df['avg_transaction_value'].iloc[-1],
                'avg_transaction_fee': onchain_df['avg_transaction_fee'].iloc[-1],
                'mempool_size': onchain_df['mempool_size'].iloc[-1],
                'active_addresses_trend': onchain_df['active_addresses'],
                'transaction_count_trend': onchain_df['transaction_count'],
                'prices': prices_df['price']
            }
            
            # Calculate network health score (0-100)
            addr_score = min(100, onchain_df['active_addresses'].iloc[-1] / 1000000 * 100)
            tx_score = min(100, onchain_df['transaction_count'].iloc[-1] / 300000 * 100)
            hash_score = min(100, onchain_df['hash_rate'].iloc[-1] / 350000000 * 100)
            network['health_score'] = (addr_score + tx_score + hash_score) / 3
            
            # Generate network insights
            network['insights'] = []
            if current_metrics['active_addresses_change'] > 10:
                network['insights'].append("Strong growth in active addresses indicates increasing network adoption")
            if current_metrics['transaction_count_change'] > 10:
                network['insights'].append("Rising transaction count suggests increased network activity")
            if current_metrics['hash_rate_change'] > 5:
                network['insights'].append("Growing hash rate indicates strong network security")
            if current_metrics['avg_transaction_fee_change'] > 20:
                network['insights'].append("Rising fees suggest high demand for block space")
            
            # HODL analysis
            hodl = {
                'dates': onchain_df.index,
                'lth_supply': onchain_df['lth_supply'].iloc[-1],
                'lth_supply_change': current_metrics['lth_supply_change'],
                'sth_supply': onchain_df['sth_supply'].iloc[-1],
                'sth_supply_change': current_metrics['sth_supply_change'],
                'waves': {
                    '> 5 years': onchain_df['lth_supply'] * 0.4,
                    '3-5 years': onchain_df['lth_supply'] * 0.3,
                    '1-3 years': onchain_df['lth_supply'] * 0.3,
                    '6-12 months': onchain_df['sth_supply'] * 0.4,
                    '1-6 months': onchain_df['sth_supply'] * 0.3,
                    '< 1 month': onchain_df['sth_supply'] * 0.3
                }
            }
            
            # Calculate HODL statistics
            hodl['statistics'] = {
                'diamond_hands_index': onchain_df['lth_supply'].iloc[-1] / onchain_df['lth_supply'].mean() * 100,
                'diamond_hands_change': current_metrics['lth_supply_change'],
                'avg_hodl_time': onchain_df['dormancy'].iloc[-1],
                'hodl_time_change': current_metrics['dormancy_change'],
                'coin_days_destroyed': onchain_df['coin_days_destroyed'].iloc[-1],
                'liveliness': onchain_df['liveliness'].iloc[-1]
            }
            
            # Mining metrics
            mining = {
                'dates': onchain_df.index,
                'hash_rate': onchain_df['hash_rate'].iloc[-1],
                'hash_rate_change': current_metrics['hash_rate_change'],
                'revenue': onchain_df['miner_revenue'].iloc[-1],
                'revenue_change': current_metrics['miner_revenue_change'],
                'difficulty': onchain_df['difficulty'].iloc[-1],
                'difficulty_change': current_metrics['difficulty_change'],
                'hash_rate_trend': onchain_df['hash_rate'],
                'revenue_trend': onchain_df['miner_revenue'],
                'hash_ribbons': onchain_df['hash_ribbons'].iloc[-1],
                'difficulty_ribbon': onchain_df['difficulty_ribbon'].iloc[-1],
                'miner_net_position': onchain_df['miner_net_position'].iloc[-1],
                'thermocap': onchain_df['thermocap'].iloc[-1]
            }
            
            # Calculate mining profitability metrics
            mining['profitability'] = {
                'breakeven_price': current_metrics['miner_revenue'] / 144,  # Daily blocks
                'breakeven_change': current_metrics['miner_revenue_change'],
                'revenue_per_th': current_metrics['miner_revenue'] / current_metrics['hash_rate'],
                'revenue_change': current_metrics['miner_revenue_change'],
                'security_score': min(100, hash_score * 0.7 + (current_metrics['miner_revenue'] / 20000000) * 30)
            }
            
            # Exchange flow analysis
            flows = {
                'dates': onchain_df.index,
                'inflow': onchain_df['exchange_inflow'],
                'outflow': -onchain_df['exchange_outflow'],  # Negative for better visualization
                'net_flow': onchain_df['exchange_inflow'].iloc[-1] - onchain_df['exchange_outflow'].iloc[-1],
                'net_flow_change': current_metrics['exchange_inflow_change'] - current_metrics['exchange_outflow_change'],
                'exchange_balance': onchain_df['exchange_balance'].iloc[-1],
                'exchange_balance_change': current_metrics['exchange_balance_change'],
                'whale_ratio': onchain_df['exchange_whale_ratio'].iloc[-1],
                'stablecoin_ratio': onchain_df['stablecoin_supply_ratio'].iloc[-1]
            }
            
            # Calculate exchange statistics
            flows['statistics'] = {
                'supply_ratio': onchain_df['exchange_balance'].iloc[-1] / 21e6 * 100,  # % of total supply
                'supply_ratio_change': current_metrics['exchange_balance_change'],
                'accumulation_score': 100 - (onchain_df['exchange_balance'].iloc[-1] / onchain_df['exchange_balance'].max() * 100),
                'accumulation_change': -current_metrics['exchange_balance_change'],
                'whale_dominance': onchain_df['exchange_whale_ratio'].iloc[-1] * 100,
                'futures_liquidations': onchain_df['futures_long_liquidations'].iloc[-1] + onchain_df['futures_short_liquidations'].iloc[-1]
            }
            
            # Market valuation metrics
            valuation = {
                'mvrv_zscore': (current_metrics['mvrv'] - onchain_df['mvrv'].mean()) / onchain_df['mvrv'].std(),
                'mvrv_zscore_change': ((current_metrics['mvrv'] - onchain_df['mvrv'].iloc[-30]) / onchain_df['mvrv'].iloc[-30] * 100) if len(onchain_df) >= 30 else 0,
                'stock_to_flow': 21000000 / (6.25 * 144 * 365),  # Current supply / yearly production
                'stock_to_flow_change': 0,  # Constant until next halving
                'puell_multiple': current_metrics['puell_multiple'],
                'puell_multiple_change': current_metrics['puell_multiple_change'],
                'nupl': (current_metrics['market_cap'] - current_metrics['realized_cap']) / current_metrics['market_cap'],
                'nupl_change': ((current_metrics['market_cap'] - current_metrics['realized_cap']) / current_metrics['market_cap'] - 
                              (onchain_df['market_cap'].iloc[-30] - onchain_df['realized_cap'].iloc[-30]) / onchain_df['market_cap'].iloc[-30]) * 100 if len(onchain_df) >= 30 else 0,
                'nvt_ratio': current_metrics['nvt_ratio'],
                'reserve_risk': current_metrics['reserve_risk'],
                'sopr': current_metrics['sopr']
            }
            
            # Add market signals
            for metric, value in valuation.items():
                if metric == 'mvrv_zscore':
                    valuation[f"{metric}_signal"] = "Bearish" if value > 7 else "Bullish" if value < -1 else "Neutral"
                elif metric == 'stock_to_flow':
                    valuation[f"{metric}_signal"] = "Bullish" if value > 50 else "Bearish" if value < 25 else "Neutral"
                elif metric == 'puell_multiple':
                    valuation[f"{metric}_signal"] = "Bearish" if value > 4 else "Bullish" if value < 0.5 else "Neutral"
                elif metric == 'nupl':
                    valuation[f"{metric}_signal"] = "Bearish" if value > 0.75 else "Bullish" if value < 0 else "Neutral"
                elif metric == 'sopr':
                    valuation[f"{metric}_signal"] = "Bearish" if value > 1.5 else "Bullish" if value < 0.95 else "Neutral"
            
            # Market cycle analysis
            price_ma = prices_df['price'].rolling(200).mean()
            current_price = prices_df['price'].iloc[-1]
            
            # Determine cycle phase
            if current_price > price_ma.iloc[-1] and current_metrics['mvrv'] > 1:
                cycle_phase = "Bull Market"
            elif current_price < price_ma.iloc[-1] and current_metrics['mvrv'] < 1:
                cycle_phase = "Bear Market"
            elif current_price > price_ma.iloc[-1] and current_metrics['mvrv'] < 1:
                cycle_phase = "Early Bull"
            else:
                cycle_phase = "Late Bear"
            
            cycle = {
                'current_phase': cycle_phase,
                'phase_duration': "180",  # Simulated value
                'cycle_progress': min(100, max(0, (current_metrics['mvrv'] - 0.5) / 3.5 * 100)),  # Based on historical MVRV range
                'cycle_prediction': "Mid-cycle" if 30 < (current_metrics['mvrv'] - 0.5) / 3.5 * 100 < 70 else 
                                  "Early-cycle" if (current_metrics['mvrv'] - 0.5) / 3.5 * 100 <= 30 else "Late-cycle"
            }
            
            # Risk metrics
            volatility = prices_df['price'].pct_change().std() * np.sqrt(252) * 100
            risk = {
                'market_risk': min(100, max(0, valuation['mvrv_zscore'] * 10 + 50)),
                'volatility_risk': min(100, volatility),
                'liquidity_risk': min(100, abs(flows['net_flow']) / 1000),
                'risk_signal': "High Risk" if valuation['mvrv_zscore'] > 5 else "Low Risk" if valuation['mvrv_zscore'] < -0.5 else "Moderate Risk",
                'volatility_change': ((prices_df['price'].pct_change().std() * np.sqrt(252) * 100) - 
                                    (prices_df['price'].shift(30).pct_change().std() * np.sqrt(252) * 100)) if len(prices_df) >= 30 else 0,
                'liquidity_change': current_metrics['exchange_balance_change']
            }
            
            # Generate market summary
            if risk['market_risk'] > 70:
                summary = "Market showing signs of overvaluation with elevated risk metrics. Consider taking profits."
            elif risk['market_risk'] < 30:
                summary = "Market appears undervalued with low risk metrics. Potential accumulation opportunity."
            else:
                summary = "Market in equilibrium with moderate risk levels. Monitor for trend confirmation."
            
            return {
                'current': current_metrics,
                'network': network,
                'hodl': hodl,
                'mining': mining,
                'flows': flows,
                'valuation': valuation,
                'cycle': cycle,
                'risk': risk,
                'summary': summary
            }
            
        except Exception as e:
            print(f"Error analyzing on-chain metrics: {str(e)}")
            return {}
            
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