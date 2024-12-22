"""Database module for handling SQLite operations."""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

from config.settings import DATA_DIR, DATABASE_PATH

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

class Database:
    def __init__(self):
        """Initialize database connection and create tables."""
        self.conn = sqlite3.connect(str(DATABASE_PATH))
        self.create_tables()
        self.initialize_festivals()

    def check_data_exists(self) -> bool:
        """Check if the database already has data."""
        cursor = self.conn.cursor()
        
        # Check festivals table
        cursor.execute('SELECT COUNT(*) FROM festivals')
        festivals_count = cursor.fetchone()[0]
        
        # Check if we have data up to 2030
        cursor.execute("SELECT COUNT(*) FROM festivals WHERE strftime('%Y', start_date) = '2030'")
        has_future_data = cursor.fetchone()[0] > 0
        
        return festivals_count > 0 and has_future_data

    def create_tables(self):
        """Create necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Create Bitcoin prices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bitcoin_prices (
                date TEXT PRIMARY KEY,
                price REAL
            )
        ''')
        
        # Create festivals table with proper date columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS festivals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL
            )
        ''')
        
        # Create NUPL data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nupl_data (
                date TEXT PRIMARY KEY,
                nupl REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indices for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_start_date ON festivals(start_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON festivals(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nupl_date ON nupl_data(date)')
        
        self.conn.commit()

    def initialize_festivals(self):
        """Initialize the festivals table with data from config."""
        # Check if database already has complete data
        if self.check_data_exists():
            return
        
        # Get festivals from config
        from config.festivals import FESTIVALS
        
        # Convert to DataFrame
        festivals_df = pd.DataFrame(FESTIVALS)
        
        # Convert dates to datetime for proper sorting
        for col in ['start_date', 'end_date']:
            festivals_df[col] = pd.to_datetime(festivals_df[col])
        
        # Sort by start date
        festivals_df = festivals_df.sort_values('start_date')
        
        # Store in database
        self.store_festivals(festivals_df)
        self.conn.commit()

    def store_bitcoin_prices(self, df: pd.DataFrame):
        """Store Bitcoin price data in the database."""
        # Convert timezone-aware dates to naive UTC
        df.index = df.index.tz_localize(None)
        df.to_sql('bitcoin_prices', self.conn, if_exists='replace', index=True, index_label='date')

    def store_festivals(self, festivals_df: pd.DataFrame):
        """Store festival data in the database."""
        # Ensure dates are in the correct format
        for col in ['start_date', 'end_date']:
            if pd.api.types.is_datetime64_any_dtype(festivals_df[col]):
                festivals_df[col] = festivals_df[col].dt.strftime('%Y-%m-%d')
        
        # Use replace to avoid duplicates
        festivals_df.to_sql('festivals', self.conn, if_exists='replace', index=False)

    def get_bitcoin_prices(self, start_date=None, end_date=None) -> pd.DataFrame:
        """Retrieve Bitcoin prices from the database."""
        # Convert input dates to naive UTC if they're datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.replace(tzinfo=None).strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.replace(tzinfo=None).strftime('%Y-%m-%d')
            
        query = 'SELECT * FROM bitcoin_prices'
        if start_date and end_date:
            query += f" WHERE date BETWEEN '{start_date}' AND '{end_date}'"
        df = pd.read_sql_query(query, self.conn, index_col='date', parse_dates=['date'])
        # Ensure index is timezone naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def get_festivals(self, category=None) -> pd.DataFrame:
        """Retrieve festival data from the database."""
        query = 'SELECT * FROM festivals'
        if category:
            query += f" WHERE category = '{category}'"
        df = pd.read_sql_query(query, self.conn, parse_dates=['start_date', 'end_date'])
        # Ensure dates are timezone-naive and in string format
        for col in ['start_date', 'end_date']:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        return df

    def get_upcoming_festivals(self) -> pd.DataFrame:
        """Get festivals that haven't started yet."""
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        query = f"""
            SELECT * FROM festivals 
            WHERE date(start_date) >= date('{today}')
            ORDER BY date(start_date)
        """
        df = pd.read_sql_query(query, self.conn, parse_dates=['start_date', 'end_date'])
        
        # Ensure dates are in string format
        for col in ['start_date', 'end_date']:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
        
        return df

    def store_nupl_data(self, df: pd.DataFrame):
        """Store NUPL data in the database.
        
        Args:
            df: DataFrame with NUPL data (index: date, columns: nupl)
        """
        # Convert index to date string if it's datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = df.index.strftime('%Y-%m-%d')
        
        # Add last_updated column with current timestamp
        df['last_updated'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            # Store in database
            df.to_sql('nupl_data', self.conn, if_exists='replace', index=True, index_label='date')
            
            # Create index after storing data
            cursor = self.conn.cursor()
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nupl_date ON nupl_data(date)')
            self.conn.commit()
            
        except Exception as e:
            print(f"Error storing NUPL data: {str(e)}")
            self.conn.rollback()

    def get_nupl_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        """Retrieve NUPL data from the database.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with NUPL data
        """
        query = 'SELECT date, nupl FROM nupl_data'
        if start_date and end_date:
            query += f" WHERE date BETWEEN '{start_date}' AND '{end_date}'"
        df = pd.read_sql_query(query, self.conn, index_col='date', parse_dates=['date'])
        return df

    def get_last_nupl_update(self) -> pd.Timestamp:
        """Get the timestamp of the last NUPL data update.
        
        Returns:
            Timestamp of last update or None if no data exists
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT MAX(last_updated) FROM nupl_data')
            result = cursor.fetchone()[0]
            return pd.Timestamp(result) if result else None
        except sqlite3.OperationalError:  # Table doesn't exist
            return None

    def __del__(self):
        """Close database connection when object is destroyed."""
        self.conn.close() 