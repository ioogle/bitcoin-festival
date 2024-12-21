"""Visualization utilities for creating charts and graphs."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List

def create_price_chart(prices_df: pd.DataFrame, title: str = "Bitcoin Price History") -> go.Figure:
    """Create a line chart of Bitcoin prices.
    
    Args:
        prices_df: DataFrame with Bitcoin prices
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices_df.index,
        y=prices_df['price'],
        name='Bitcoin Price',
        line=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_festival_performance_chart(stats_df: pd.DataFrame) -> go.Figure:
    """Create a bar chart comparing festival performance.
    
    Args:
        stats_df: DataFrame with festival statistics
        
    Returns:
        Plotly figure object
    """
    fig = px.bar(
        stats_df,
        x='Festival',
        y='Price Change %',
        color='Price Change %',
        title='Festival Performance Comparison',
        color_continuous_scale=['red', 'yellow', 'green']
    )
    
    fig.update_layout(
        xaxis_title='Festival',
        yaxis_title='Price Change (%)',
        showlegend=False
    )
    
    return fig

def create_yearly_comparison_chart(stats_df: pd.DataFrame, category: str) -> go.Figure:
    """Create a chart comparing festival performance across years.
    
    Args:
        stats_df: DataFrame with festival statistics
        category: Festival category to analyze
        
    Returns:
        Plotly figure object
    """
    category_stats = stats_df[stats_df['Category'] == category].copy()
    category_stats['Year'] = pd.to_datetime(category_stats['Start Date']).dt.year
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Change by Year', 'Price Levels by Year'),
        vertical_spacing=0.15
    )
    
    # Price change chart
    fig.add_trace(
        go.Bar(
            x=category_stats['Year'],
            y=category_stats['Price Change %'],
            name='Price Change %',
            marker_color=category_stats['Price Change %'].apply(
                lambda x: 'green' if x > 0 else 'red'
            )
        ),
        row=1, col=1
    )
    
    # Price levels chart
    fig.add_trace(
        go.Bar(
            x=category_stats['Year'],
            y=category_stats['Start Price'],
            name='Start Price',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=category_stats['Year'],
            y=category_stats['End Price'],
            name='End Price',
            marker_color='darkblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{category} - Yearly Comparison',
        height=800,
        showlegend=True,
        barmode='group'
    )
    
    return fig

def create_volatility_chart(stats_df: pd.DataFrame) -> go.Figure:
    """Create a chart showing price volatility during festivals.
    
    Args:
        stats_df: DataFrame with festival statistics
        
    Returns:
        Plotly figure object
    """
    fig = px.scatter(
        stats_df,
        x='Volatility',
        y='Price Change %',
        color='Category',
        size='Max Return %',
        hover_data=['Festival', 'Start Date', 'End Date'],
        title='Festival Volatility vs Returns'
    )
    
    fig.update_layout(
        xaxis_title='Volatility (%)',
        yaxis_title='Price Change (%)',
        showlegend=True
    )
    
    return fig

def format_metrics(value: float, is_price: bool = False) -> str:
    """Format numeric values for display.
    
    Args:
        value: Number to format
        is_price: Whether the value is a price
        
    Returns:
        Formatted string
    """
    if is_price:
        return f"${value:,.2f}"
    return f"{value:,.1f}%" 