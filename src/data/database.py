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
        self.reset_database()  # Reset database on startup
        self.create_tables()
        self.initialize_festivals()

    def reset_database(self):
        """Reset the database by dropping all tables."""
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS festivals")
        cursor.execute("DROP TABLE IF EXISTS bitcoin_prices")
        self.conn.commit()

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
        
        # Create indices for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_start_date ON festivals(start_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON festivals(category)')
        
        self.conn.commit()

    def initialize_festivals(self):
        """Initialize the festivals table with data from config."""
        # Check if festivals table is empty
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM festivals')
        count = cursor.fetchone()[0]
        
        if count == 0:
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

    def __del__(self):
        """Close database connection when object is destroyed."""
        self.conn.close() 