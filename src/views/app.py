"""Main Streamlit application for Bitcoin Festival Price Tracker."""

import sys
from pathlib import Path
from typing import Dict

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.absolute())
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from data.database import Database
from data.fetcher import DataFetcher
from utils.visualization import (
    create_price_chart,
    create_festival_performance_chart,
    create_yearly_comparison_chart,
    create_volatility_chart,
    format_metrics
)

class BitcoinFestivalApp:
    def __init__(self):
        """Initialize the application."""
        self.db = Database()
        self.fetcher = DataFetcher()
        self.setup_page()
        self.load_initial_data()

    def setup_page(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Bitcoin Festival Price Tracker",
            page_icon="📈",
            layout="wide"
        )
        st.title("🎊 Bitcoin Festival Price Tracker")

    def load_initial_data(self):
        """Load initial data into the database."""
        # Load Bitcoin prices if not exists
        prices = self.db.get_bitcoin_prices()
        if prices.empty:
            with st.spinner("Fetching Bitcoin price data..."):
                prices = self.fetcher.fetch_bitcoin_prices()
                if not prices.empty:
                    self.db.store_bitcoin_prices(prices)
                else:
                    st.error("Failed to fetch Bitcoin price data.")
                    st.stop()
        
        # Load festivals if not exists
        festivals = self.db.get_festivals()
        if len(festivals) == 0:
            festivals_df = self.fetcher.get_festivals_data()
            self.db.store_festivals(festivals_df)

    def show_overview(self, prices_df: pd.DataFrame, stats_df: pd.DataFrame):
        """Show overview of Bitcoin price performance during festivals."""
        st.header("📊 Overview")
        
        # Price chart
        st.plotly_chart(create_price_chart(prices_df), use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Performance Summary")
        
        # Calculate overall statistics
        avg_return = stats_df['Price Change %'].mean()
        success_rate = (stats_df['Price Change %'] > 0).mean() * 100
        avg_volatility = stats_df['Volatility'].mean()
        avg_duration = stats_df['Duration'].mean()
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Average Return",
                format_metrics(avg_return)
            )
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%"
            )
        with col2:
            st.metric(
                "Average Volatility",
                format_metrics(avg_volatility)
            )
            st.metric(
                "Best Return",
                format_metrics(stats_df['Price Change %'].max())
            )
        with col3:
            st.metric(
                "Worst Return",
                format_metrics(stats_df['Price Change %'].min())
            )
            st.metric(
                "Average Festival Duration",
                f"{avg_duration:.1f} days"
            )
        
        # Festival performance chart
        st.subheader("🎯 Festival Performance")
        st.plotly_chart(create_festival_performance_chart(stats_df), use_container_width=True)

    def show_drawdown_analysis(self, prices_df: pd.DataFrame, stats_df: pd.DataFrame, time_range: str = None):
        """Show drawdown and volatility analysis."""
        st.header("📉 Drawdown & Volatility Analysis")
        
        # Calculate drawdown metrics
        rolling_drawdown = self.fetcher.calculate_drawdown(prices_df['price'])
        max_drawdown = rolling_drawdown.min()
        current_drawdown = rolling_drawdown.iloc[-1]
        
        # Calculate drawdown statistics
        drawdown_stats = pd.DataFrame({
            'Metric': [
                'Maximum Drawdown',
                'Current Drawdown',
                'Average Drawdown',
                'Drawdown Std Dev',
                'Time in Drawdown',
                'Recovery Time (Avg)'
            ],
            'Value': [
                format_metrics(max_drawdown),
                format_metrics(current_drawdown),
                format_metrics(rolling_drawdown.mean()),
                format_metrics(rolling_drawdown.std()),
                f"{(rolling_drawdown < 0).mean():.1%}",
                f"{rolling_drawdown[rolling_drawdown < 0].count() / (rolling_drawdown < 0).sum():.1f} days"
            ]
        })
        
        # Market Health Overview
        st.subheader("🏥 Market Health Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Current Drawdown",
                format_metrics(current_drawdown)
            )
            st.metric(
                "Maximum Drawdown",
                format_metrics(max_drawdown)
            )
        with col2:
            st.metric(
                "Average Drawdown",
                format_metrics(rolling_drawdown.mean())
            )
            st.metric(
                "Drawdown Volatility",
                format_metrics(rolling_drawdown.std())
            )
        with col3:
            st.metric(
                "Time in Drawdown",
                f"{(rolling_drawdown < 0).mean():.1%}"
            )
            st.metric(
                "Recovery Time",
                f"{rolling_drawdown[rolling_drawdown < 0].count() / (rolling_drawdown < 0).sum():.1f} days"
            )
        
        # Drawdown Timeline
        st.subheader("📈 Drawdown Timeline")
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=prices_df.index,
                y=prices_df['price'],
                name='Bitcoin Price',
                line=dict(color='blue', width=1)
            ),
            secondary_y=True
        )
        
        # Add drawdown area
        fig.add_trace(
            go.Scatter(
                x=rolling_drawdown.index,
                y=rolling_drawdown,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            ),
            secondary_y=False
        )
        
        # Update layout
        fig.update_layout(
            title=f'Bitcoin Price and Drawdown History ({time_range})',
            xaxis_title='Date',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            hovermode='x unified'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Drawdown %", secondary_y=False)
        fig.update_yaxes(title_text="Bitcoin Price (USD)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown Distribution Analysis
        st.subheader("📊 Drawdown Distribution")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # Create drawdown distribution chart
            fig_dist = go.Figure()
            fig_dist.add_trace(
                go.Histogram(
                    x=rolling_drawdown,
                    nbinsx=50,
                    name='Drawdown Distribution'
                )
            )
            fig_dist.update_layout(
                title='Distribution of Drawdowns',
                xaxis_title='Drawdown %',
                yaxis_title='Frequency',
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.dataframe(drawdown_stats, use_container_width=True)
        
        # Weekly Volatility Analysis
        st.subheader("📊 Weekly Volatility Analysis")
        
        # Calculate weekly volatility metrics
        weekly_stats = self.fetcher.calculate_weekly_volatility(prices_df)
        period_stats = weekly_stats
        
        # Weekly volatility summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Average Weekly Volatility",
                format_metrics(period_stats['Volatility'].mean())
            )
            st.metric(
                "Volatility Trend",
                format_metrics(
                    period_stats['Volatility'].tail(4).mean() - 
                    period_stats['Volatility'].head(4).mean()
                )
            )
        with col2:
            st.metric(
                "Maximum Weekly Volatility",
                format_metrics(period_stats['Volatility'].max())
            )
            st.metric(
                "90th Percentile Volatility",
                format_metrics(period_stats['Volatility'].quantile(0.9))
            )
        with col3:
            st.metric(
                "Current Weekly Volatility",
                format_metrics(period_stats['Volatility'].iloc[-1])
            )
            st.metric(
                "Volatility of Volatility",
                format_metrics(period_stats['Volatility'].std())
            )
        
        # Create tabs for different volatility visualizations
        vol_tab1, vol_tab2, vol_tab3 = st.tabs(["📈 Time Series", "📊 Distribution", "🔄 Cycles"])
        
        with vol_tab1:
            # Enhanced time series chart
            fig_vol = go.Figure()
            
            # Add volatility line
            fig_vol.add_trace(
                go.Scatter(
                    x=weekly_stats.index,
                    y=weekly_stats['Volatility'],
                    name='Weekly Volatility',
                    line=dict(color='blue')
                )
            )
            
            # Add range band
            fig_vol.add_trace(
                go.Scatter(
                    x=weekly_stats.index,
                    y=weekly_stats['Range %'],
                    name='Price Range %',
                    line=dict(color='gray', dash='dash')
                )
            )
            
            # Add returns
            fig_vol.add_trace(
                go.Scatter(
                    x=weekly_stats.index,
                    y=weekly_stats['Return %'],
                    name='Weekly Return',
                    line=dict(color='green')
                )
            )
            
            fig_vol.update_layout(
                title=f'Weekly Volatility Metrics ({time_range})',
                xaxis_title='Date',
                yaxis_title='Percentage',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with vol_tab2:
            # Create side-by-side box and violin plots
            fig_dist = go.Figure()
            
            # Add box plot
            fig_dist.add_trace(
                go.Box(
                    y=weekly_stats['Volatility'],
                    name='Volatility',
                    boxpoints='outliers'
                )
            )
            
            # Add violin plot
            fig_dist.add_trace(
                go.Violin(
                    y=weekly_stats['Volatility'],
                    name='Distribution',
                    side='positive',
                    line_color='blue'
                )
            )
            
            fig_dist.update_layout(
                title='Volatility Distribution Analysis',
                yaxis_title='Weekly Volatility %',
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with vol_tab3:
            # Volatility cycle analysis
            cycle_data = pd.DataFrame({
                'Month': weekly_stats.index.month,
                'Volatility': weekly_stats['Volatility']
            })
            monthly_vol = cycle_data.groupby('Month')['Volatility'].agg(['mean', 'std', 'count'])
            
            # Create monthly seasonality chart
            fig_cycle = go.Figure()
            
            # Add mean volatility line
            fig_cycle.add_trace(
                go.Scatter(
                    x=monthly_vol.index,
                    y=monthly_vol['mean'],
                    name='Average Volatility',
                    line=dict(color='blue')
                )
            )
            
            # Add confidence bands
            fig_cycle.add_trace(
                go.Scatter(
                    x=monthly_vol.index,
                    y=monthly_vol['mean'] + monthly_vol['std'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                )
            )
            
            fig_cycle.add_trace(
                go.Scatter(
                    x=monthly_vol.index,
                    y=monthly_vol['mean'] - monthly_vol['std'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='±1 Std Dev'
                )
            )
            
            fig_cycle.update_layout(
                title='Monthly Volatility Seasonality',
                xaxis_title='Month',
                yaxis_title='Volatility %',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_cycle, use_container_width=True)
        
        # Detailed Statistics
        with st.expander("📊 Detailed Weekly Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                weekly_detailed_stats = pd.DataFrame({
                    'Metric': [
                        'Weeks Analyzed',
                        'Positive Return Weeks',
                        'Negative Return Weeks',
                        'Average Positive Return',
                        'Average Negative Return',
                        'Maximum Weekly Gain',
                        'Maximum Weekly Loss'
                    ],
                    'Value': [
                        f"{len(period_stats)} weeks",
                        f"{(period_stats['Return %'] > 0).mean():.1%}",
                        f"{(period_stats['Return %'] < 0).mean():.1%}",
                        format_metrics(period_stats[period_stats['Return %'] > 0]['Return %'].mean()),
                        format_metrics(period_stats[period_stats['Return %'] < 0]['Return %'].mean()),
                        format_metrics(period_stats['Return %'].max()),
                        format_metrics(period_stats['Return %'].min())
                    ]
                })
                st.dataframe(weekly_detailed_stats, use_container_width=True)
            
            with col2:
                volatility_detailed_stats = pd.DataFrame({
                    'Metric': [
                        'Average Volatility',
                        'Volatility Std Dev',
                        'Volatility Skew',
                        'Volatility Kurtosis',
                        'High Volatility Weeks',
                        'Low Volatility Weeks',
                        'Volatility Trend'
                    ],
                    'Value': [
                        format_metrics(period_stats['Volatility'].mean()),
                        format_metrics(period_stats['Volatility'].std()),
                        f"{period_stats['Volatility'].skew():.2f}",
                        f"{period_stats['Volatility'].kurtosis():.2f}",
                        f"{(period_stats['Volatility'] > period_stats['Volatility'].mean() + period_stats['Volatility'].std()).mean():.1%}",
                        f"{(period_stats['Volatility'] < period_stats['Volatility'].mean() - period_stats['Volatility'].std()).mean():.1%}",
                        "↑ Increasing" if period_stats['Volatility'].tail(4).mean() > period_stats['Volatility'].head(4).mean() else "↓ Decreasing"
                    ]
                })
                st.dataframe(volatility_detailed_stats, use_container_width=True)
        
        # Festival-specific Analysis
        st.subheader("🎯 Festival-specific Analysis")
        
        # Create a more detailed festival analysis table
        festival_analysis = stats_df[[
            'Festival', 'Category', 'Duration',
            'Max Drawdown %', 'Drawdown Duration',
            'Avg Weekly Volatility', 'Max Weekly Volatility',
            'Max Weekly Range %'
        ]].copy()
        
        # Add volatility ratio and recovery metrics
        festival_analysis['Volatility Ratio'] = festival_analysis['Max Weekly Volatility'] / festival_analysis['Avg Weekly Volatility']
        festival_analysis['Recovery Rate'] = abs(festival_analysis['Max Drawdown %']) / festival_analysis['Drawdown Duration']
        
        # Sort by max drawdown
        festival_analysis = festival_analysis.sort_values('Max Drawdown %')
        
        st.dataframe(
            festival_analysis.style.format({
                'Duration': '{:.0f} days',
                'Max Drawdown %': format_metrics,
                'Drawdown Duration': '{:.0f} days',
                'Avg Weekly Volatility': format_metrics,
                'Max Weekly Volatility': format_metrics,
                'Max Weekly Range %': format_metrics,
                'Volatility Ratio': '{:.2f}',
                'Recovery Rate': '{:.2f}% per day'
            }).background_gradient(
                subset=['Max Drawdown %', 'Max Weekly Volatility'],
                cmap='RdYlGn_r'
            ),
            use_container_width=True
        )
        
        # Category-wise Analysis with Enhanced Metrics
        st.subheader("📊 Category-wise Risk Analysis")
        
        category_stats = stats_df.groupby('Category').agg({
            'Volatility': ['mean', 'std', 'max'],
            'Max Drawdown %': ['mean', 'min', 'std'],
            'Drawdown Duration': ['mean', 'max'],
            'Avg Weekly Volatility': 'mean',
            'Max Weekly Volatility': 'max',
            'Max Weekly Range %': 'max'
        }).round(2)
        
        # Flatten column names
        category_stats.columns = [
            'Avg Volatility', 'Vol Std Dev', 'Max Vol',
            'Avg Drawdown', 'Max Drawdown', 'Drawdown Std Dev',
            'Avg DD Duration', 'Max DD Duration',
            'Avg Weekly Vol', 'Max Weekly Vol', 'Max Weekly Range'
        ]
        
        # Calculate risk metrics
        category_stats['Risk Score'] = (
            -category_stats['Max Drawdown'] * 0.4 +
            category_stats['Avg Weekly Vol'] * 0.3 +
            category_stats['Drawdown Std Dev'] * 0.3
        ).rank()
        
        st.dataframe(
            category_stats.style.format({
                col: format_metrics for col in category_stats.columns
                if 'Duration' not in col and 'Score' not in col
            }).format({
                'Avg DD Duration': '{:.0f} days',
                'Max DD Duration': '{:.0f} days',
                'Risk Score': '{:.0f}'
            }).background_gradient(
                subset=['Risk Score'],
                cmap='RdYlGn_r'
            ),
            use_container_width=True
        )

    def show_festival_analysis(self, prices_df: pd.DataFrame, stats_df: pd.DataFrame):
        """Show detailed festival analysis."""
        st.header("🎯 Festival Analysis")
        
        # Festival selector
        categories = sorted(stats_df['Category'].unique())
        selected_category = st.selectbox(
            "Select Festival",
            options=['All'] + categories
        )
        
        if selected_category == 'All':
            # Show volatility analysis
            st.plotly_chart(create_volatility_chart(stats_df), use_container_width=True)
            
            # Show detailed statistics
            st.dataframe(
                stats_df.style.format({
                    'Start Price': lambda x: format_metrics(x, True),
                    'End Price': lambda x: format_metrics(x, True),
                    'Min Price': lambda x: format_metrics(x, True),
                    'Max Price': lambda x: format_metrics(x, True),
                    'Price Change %': format_metrics,
                    'Max Return %': format_metrics,
                    'Volatility': format_metrics
                })
            )
        else:
            # Show yearly comparison
            st.plotly_chart(
                create_yearly_comparison_chart(stats_df, selected_category),
                use_container_width=True
            )
            
            # Show category statistics
            category_stats = stats_df[stats_df['Category'] == selected_category]
            st.dataframe(
                category_stats.style.format({
                    'Start Price': lambda x: format_metrics(x, True),
                    'End Price': lambda x: format_metrics(x, True),
                    'Min Price': lambda x: format_metrics(x, True),
                    'Max Price': lambda x: format_metrics(x, True),
                    'Price Change %': format_metrics,
                    'Max Return %': format_metrics,
                    'Volatility': format_metrics
                })
            )

    def show_upcoming_festivals(self, prices_df: pd.DataFrame, end_date: pd.Timestamp = None):
        """Show upcoming festivals with predictions and scenarios."""
        st.header("🔮 Upcoming Festivals")
        
        # Calculate current Bitcoin price and trends
        current_price = prices_df['price'].iloc[-1]
        price_30d_change = ((current_price - prices_df['price'].iloc[-30]) / prices_df['price'].iloc[-30]) * 100
        price_90d_change = ((current_price - prices_df['price'].iloc[-90]) / prices_df['price'].iloc[-90]) * 100
        
        # Show current market context
        st.subheader("Current Market Context")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Bitcoin Price", format_metrics(current_price, True))
        with col2:
            st.metric("30-Day Change", format_metrics(price_30d_change))
        with col3:
            st.metric("90-Day Change", format_metrics(price_90d_change))
        
        # Get upcoming festivals within the selected date range
        now = pd.Timestamp.now()
        end_date = end_date or pd.Timestamp('2030-12-31')
        
        upcoming = self.db.get_festivals()
        upcoming = upcoming[
            (pd.to_datetime(upcoming['start_date']) > now) &
            (pd.to_datetime(upcoming['start_date']) <= end_date)
        ]
        upcoming = upcoming.sort_values('start_date')
        
        if upcoming.empty:
            st.warning("⚠️ No festivals found in the selected prediction horizon.")
            return
        
        # Show timeline of upcoming festivals
        st.subheader("📅 Festival Timeline")
        timeline_data = []
        
        for _, festival in upcoming.iterrows():
            start_date = pd.to_datetime(festival['start_date'])
            days_until = (start_date - now).days
            timeline_data.append({
                'Festival': festival['name'],
                'Category': festival['category'],
                'Start Date': festival['start_date'],
                'End Date': festival['end_date'],
                'Days Until': days_until
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(
            timeline_df.style.format({
                'Days Until': '{:,.0f} days'
            }),
            use_container_width=True
        )
        
        # Analyze each upcoming festival
        for _, festival in upcoming.iterrows():
            st.markdown("---")
            st.subheader(festival['name'])
            
            # Get historical performance
            historical = self.db.get_festivals()
            historical = historical[
                (pd.to_datetime(historical['start_date']) < pd.to_datetime(festival['start_date'])) &
                (historical['category'] == festival['category'])
            ]
            
            if not historical.empty:
                stats = self.fetcher.analyze_festival_performance(prices_df, historical)
                
                # Basic festival information
                col1, col2, col3 = st.columns(3)
                with col1:
                    start_date = pd.to_datetime(festival['start_date'])
                    days_until = (start_date - now).days
                    st.metric(
                        "Festival Period",
                        f"{festival['start_date']} to {festival['end_date']}",
                        f"In {days_until} days"
                    )
                with col2:
                    avg_return = stats['Price Change %'].mean()
                    st.metric(
                        "Historical Avg Return",
                        format_metrics(avg_return)
                    )
                with col3:
                    success_rate = (stats['Price Change %'] > 0).mean() * 100
                    st.metric(
                        "Historical Success Rate",
                        f"{success_rate:.0f}%"
                    )
                
                # Scenario Analysis
                st.subheader(" Scenario Analysis")
                
                # Calculate scenario metrics
                avg_volatility = stats['Volatility'].mean()
                max_return = stats['Max Return %'].max()
                min_return = stats['Price Change %'].min()
                median_return = stats['Price Change %'].median()
                
                scenarios = {
                    "Bullish Scenario": {
                        "Return": max(avg_return * 1.5, max_return),
                        "Description": "Based on historical best performance and current market momentum",
                        "Confidence": min(success_rate, 100) if price_90d_change > 0 else success_rate * 0.8
                    },
                    "Base Scenario": {
                        "Return": median_return,
                        "Description": "Based on historical median performance",
                        "Confidence": success_rate
                    },
                    "Bearish Scenario": {
                        "Return": min(avg_return * 0.5, min_return),
                        "Description": "Based on historical worst performance and market risks",
                        "Confidence": 100 - success_rate if price_90d_change < 0 else (100 - success_rate) * 0.8
                    }
                }
                
                # Display scenarios
                for scenario, data in scenarios.items():
                    expander = st.expander(f"{scenario}")
                    with expander:
                        col1, col2 = st.columns(2)
                        with col1:
                            predicted_price = current_price * (1 + data["Return"]/100)
                            st.metric(
                                "Predicted Price",
                                format_metrics(predicted_price, True),
                                format_metrics(data["Return"])
                            )
                        with col2:
                            st.metric(
                                "Confidence Level",
                                f"{data['Confidence']:.0f}%"
                            )
                        st.write(f"**Analysis:** {data['Description']}")
                        
                        # Trading suggestions
                        if data["Return"] > 0:
                            st.success(f"💡 Suggested Strategy: Consider buying before {festival['start_date']} "
                                     f"with a target price of {format_metrics(predicted_price, True)}")
                        else:
                            st.warning(f"💡 Suggested Strategy: Consider waiting or setting tight stop losses. "
                                     f"Potential downside to {format_metrics(predicted_price, True)}")
                
                # Historical Performance Chart
                st.subheader("📈 Historical Performance")
                st.plotly_chart(
                    create_yearly_comparison_chart(stats, festival['category']),
                    use_container_width=True
                )
                
                # Additional insights
                best_year = stats.loc[stats['Price Change %'].idxmax()]
                worst_year = stats.loc[stats['Price Change %'].idxmin()]
                
                with st.expander("📝 Detailed Historical Analysis"):
                    st.write("**Best Historical Performance:**")
                    st.write(f"- Year: {best_year['Festival']}")
                    st.write(f"- Return: {format_metrics(best_year['Price Change %'])}")
                    st.write(f"- Price Range: {format_metrics(best_year['Start Price'], True)} to {format_metrics(best_year['End Price'], True)}")
                    
                    st.write("\n**Worst Historical Performance:**")
                    st.write(f"- Year: {worst_year['Festival']}")
                    st.write(f"- Return: {format_metrics(worst_year['Price Change %'])}")
                    st.write(f"- Price Range: {format_metrics(worst_year['Start Price'], True)} to {format_metrics(worst_year['End Price'], True)}")
                    
                    st.write("\n**Trading Patterns:**")
                    st.write(f"- Average Volatility: {format_metrics(avg_volatility)}")
                    st.write(f"- Best Buy Day: Typically {stats['Best Buy Date'].mode().iloc[0] if not stats['Best Buy Date'].mode().empty else 'varies'}")
                    st.write(f"- Best Sell Day: Typically {stats['Best Sell Date'].mode().iloc[0] if not stats['Best Sell Date'].mode().empty else 'varies'}")
            else:
                st.info(f"No historical data available for analysis of {festival['category']}.")

    def show_bull_bear_analysis(self, prices_df: pd.DataFrame):
        """Show bull and bear market analysis with halving cycle correlation."""
        st.header("🐂🐻 Bull & Bear Market Analysis")
        
        # Identify market cycles
        cycles_df = self.fetcher.identify_market_cycles(prices_df)
        
        if cycles_df.empty:
            st.warning("No market cycles found in the selected date range.")
            return
        
        # Market Cycles Overview
        st.subheader("📊 Market Cycles Overview")
        
        # Summary metrics
        total_bull_days = cycles_df[cycles_df['Type'] == 'Bull']['Duration'].sum()
        total_bear_days = cycles_df[cycles_df['Type'] == 'Bear']['Duration'].sum()
        total_days = total_bull_days + total_bear_days
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Cycles",
                f"{len(cycles_df)} cycles"
            )
            st.metric(
                "Average Cycle Length",
                f"{total_days / len(cycles_df):.0f} days"
            )
        with col2:
            st.metric(
                "Time in Bull Market",
                f"{(total_bull_days / total_days * 100):.1f}%"
            )
            st.metric(
                "Average Bull Duration",
                f"{cycles_df[cycles_df['Type'] == 'Bull']['Duration'].mean():.0f} days"
            )
        with col3:
            st.metric(
                "Time in Bear Market",
                f"{(total_bear_days / total_days * 100):.1f}%"
            )
            st.metric(
                "Average Bear Duration",
                f"{cycles_df[cycles_df['Type'] == 'Bear']['Duration'].mean():.0f} days"
            )
        
        # Market Cycles Table
        st.dataframe(
            cycles_df.style.format({
                'Duration': '{:.0f} days',
                'Return %': format_metrics,
                'Start Price': lambda x: format_metrics(x, True),
                'End Price': lambda x: format_metrics(x, True)
            }),
            use_container_width=True
        )
        
        # Major Market Events Analysis
        st.subheader("🎯 Major Market Events")
        
        # Get major events and analyze their impact
        events = self.fetcher.get_major_events()
        event_impacts = []
        
        for event in events:
            impact = self.fetcher.analyze_event_impact(prices_df, event)
            if impact:
                event_impacts.append(impact)
        
        if event_impacts:
            # Create DataFrame for event impacts
            events_df = pd.DataFrame(event_impacts)
            
            # Display events table
            st.dataframe(
                events_df[[
                    'date', 'event', 'description', 'type',
                    'pre_event_return', 'post_event_return',
                    'pre_event_volatility', 'post_event_volatility'
                ]].style.format({
                    'pre_event_return': format_metrics,
                    'post_event_return': format_metrics,
                    'pre_event_volatility': format_metrics,
                    'post_event_volatility': format_metrics
                }),
                use_container_width=True
            )
        
        # Historical Price Chart with Events and Halvings
        st.subheader("📈 Price History with Major Events")
        
        # Create price chart
        fig = px.line(
            prices_df,
            x=prices_df.index,
            y='price',
            title='Bitcoin Price with Major Events and Halvings'
        )
        
        # Add vertical lines for halving events
        for halving_date in self.fetcher.get_halving_dates():
            if halving_date >= prices_df.index[0] and halving_date <= prices_df.index[-1]:
                fig.add_shape(
                    type="line",
                    x0=halving_date,
                    x1=halving_date,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="orange", width=2, dash="dash")
                )
                fig.add_annotation(
                    x=halving_date,
                    y=1,
                    yref="paper",
                    text=f"Halving {halving_date.strftime('%Y-%m-%d')}",
                    showarrow=False,
                    textangle=-90,
                    yshift=10
                )
        
        # Add markers for major events
        for event in events:
            event_date = event['date']
            if event_date >= prices_df.index[0] and event_date <= prices_df.index[-1]:
                # Get price at event
                event_price = prices_df[prices_df.index <= event_date]['price'].iloc[-1]
                
                # Add marker
                fig.add_trace(
                    go.Scatter(
                        x=[event_date],
                        y=[event_price],
                        mode='markers+text',
                        name=event['event'],
                        text=[event['event']],
                        textposition='top center',
                        marker=dict(
                            size=10,
                            symbol='circle',
                            color='red' if event['type'] == 'negative' else 'green' if event['type'] == 'positive' else 'gray'
                        ),
                        showlegend=True
                    )
                )
        
        # Update layout
        fig.update_layout(
            showlegend=True,
            hovermode='x unified',
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            height=800  # Make the chart taller for better visibility
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Halving Analysis
        st.subheader("⛏️ Bitcoin Halving Analysis")
        
        # Get halving dates and analyze impact
        halving_stats = self.fetcher.analyze_halving_impact(prices_df)
        
        if halving_stats.empty:
            st.info("No halving events found in the selected date range.")
        else:
            # Display halving statistics
            st.dataframe(
                halving_stats.style.format({
                    'Pre-Halving Return %': format_metrics,
                    'Post-Halving Return %': format_metrics,
                    'Pre-Halving Volatility': format_metrics,
                    'Post-Halving Volatility': format_metrics,
                    'Price at Halving': lambda x: format_metrics(x, True) if pd.notnull(x) else 'Future',
                    'Price After 1 Year': lambda x: format_metrics(x, True) if pd.notnull(x) else 'N/A'
                }),
                use_container_width=True
            )
            
            # Halving Impact Summary (only for past halvings)
            past_halvings = halving_stats[~halving_stats['Is Future']]
            if not past_halvings.empty:
                avg_pre_return = past_halvings['Pre-Halving Return %'].mean()
                avg_post_return = past_halvings['Post-Halving Return %'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Average Pre-Halving Return",
                        format_metrics(avg_pre_return)
                    )
                    st.metric(
                        "Pre-Halving Volatility",
                        format_metrics(past_halvings['Pre-Halving Volatility'].mean())
                    )
                with col2:
                    st.metric(
                        "Average Post-Halving Return",
                        format_metrics(avg_post_return)
                    )
                    st.metric(
                        "Post-Halving Volatility",
                        format_metrics(past_halvings['Post-Halving Volatility'].mean())
                    )
                
                # Add insights about future halvings
                future_halvings = halving_stats[halving_stats['Is Future']]
                if not future_halvings.empty:
                    st.subheader("🔮 Future Halving Analysis")
                    for _, halving in future_halvings.iterrows():
                        st.write(f"**Upcoming Halving: {halving['Halving Date'].strftime('%Y-%m-%d')}**")
                        st.write(f"- Pre-halving trend: {format_metrics(halving['Pre-Halving Return %'])}")
                        st.write(f"- Current volatility: {format_metrics(halving['Pre-Halving Volatility'])}")
                        
                        # Compare with historical averages
                        if halving['Pre-Halving Return %'] > avg_pre_return:
                            st.success("🔥 Current pre-halving return is above historical average")
                        else:
                            st.warning("📉 Current pre-halving return is below historical average")
                        
                        if halving['Pre-Halving Volatility'] > past_halvings['Pre-Halving Volatility'].mean():
                            st.warning("📊 Current volatility is higher than historical average")
                        else:
                            st.success("👍 Current volatility is lower than historical average")
        
        # Next Halving Countdown
        next_halving = self.fetcher.get_next_halving_date()
        if next_halving:
            days_to_halving = (next_halving - pd.Timestamp.now()).days
            st.info(f"📅 Next Bitcoin Halving: {next_halving.strftime('%Y-%m-%d')} (in {days_to_halving} days)")

    def show_sentiment_analysis(self, prices_df: pd.DataFrame):
        """Show sentiment analysis page."""
        st.title("Market Sentiment Analysis")
        
        try:
            # Get Fear & Greed Index data
            fear_greed_df = self.fetcher.fetch_fear_greed_index(
                start_date=prices_df.index[0].strftime('%Y-%m-%d'),
                end_date=prices_df.index[-1].strftime('%Y-%m-%d')
            )
            
            if fear_greed_df.empty:
                st.warning("No sentiment data available for the selected date range.")
                return
            
            # Get current sentiment
            current_fg = fear_greed_df['value'].iloc[-1]
            current_label = self.fetcher.get_fear_greed_label(int(current_fg))
            
            # Get historical values for comparison
            week_ago_fg = fear_greed_df['value'].iloc[-7] if len(fear_greed_df) >= 7 else None
            month_ago_fg = fear_greed_df['value'].iloc[-30] if len(fear_greed_df) >= 30 else None
            
            # Current Sentiment Overview
            st.subheader("📊 Current Market Sentiment")
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Current Sentiment",
                    current_label,
                    f"Index: {current_fg:.0f}"
                )
            with col2:
                if week_ago_fg is not None:
                    weekly_change = current_fg - week_ago_fg
                    week_ago_label = self.fetcher.get_fear_greed_label(int(week_ago_fg))
                    st.metric(
                        "Weekly Change",
                        f"{weekly_change:+.0f}",
                        week_ago_label
                    )
            with col3:
                if month_ago_fg is not None:
                    monthly_change = current_fg - month_ago_fg
                    month_ago_label = self.fetcher.get_fear_greed_label(int(month_ago_fg))
                    st.metric(
                        "Monthly Change",
                        f"{monthly_change:+.0f}",
                        month_ago_label
                    )
            
            # Historical Sentiment Chart with Events and Price
            st.subheader("📈 Historical Sentiment & Major Events")
            
            # Create figure with secondary y-axis
            fig_fg = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add Fear & Greed Index line
            fig_fg.add_trace(
                go.Scatter(
                    x=fear_greed_df.index,
                    y=fear_greed_df['value'],
                    name='Fear & Greed Index',
                    line=dict(color='blue', width=2)
                ),
                secondary_y=False
            )
            
            # Add Bitcoin price line on secondary axis
            price_resampled = prices_df.resample('D')['price'].last()
            fig_fg.add_trace(
                go.Scatter(
                    x=price_resampled.index,
                    y=price_resampled.values,
                    name='Bitcoin Price',
                    line=dict(color='gray', width=1, dash='dot')
                ),
                secondary_y=True
            )
            
            # Add colored background zones for sentiment
            fig_fg.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.1, line_width=0, name="Fear")
            fig_fg.add_hrect(y0=25, y1=45, fillcolor="orange", opacity=0.1, line_width=0, name="Fear")
            fig_fg.add_hrect(y0=45, y1=55, fillcolor="yellow", opacity=0.1, line_width=0, name="Neutral")
            fig_fg.add_hrect(y0=55, y1=75, fillcolor="lightgreen", opacity=0.1, line_width=0, name="Greed")
            fig_fg.add_hrect(y0=75, y1=100, fillcolor="green", opacity=0.1, line_width=0, name="Extreme Greed")
            
            # Add major events markers
            events = self.fetcher.get_major_events()
            for event in events:
                event_date = event['date']
                if event_date >= fear_greed_df.index[0] and event_date <= fear_greed_df.index[-1]:
                    # Get sentiment and price at event
                    event_sentiment = fear_greed_df[fear_greed_df.index <= event_date]['value'].iloc[-1]
                    event_price = price_resampled[price_resampled.index <= event_date].iloc[-1]
                    
                    # Add marker
                    fig_fg.add_trace(
                        go.Scatter(
                            x=[event_date],
                            y=[event_sentiment],
                            mode='markers',
                            name=event['event'],
                            marker=dict(
                                size=12,
                                symbol='circle',
                                color='red' if event['type'] == 'negative' else 'green' if event['type'] == 'positive' else 'gray',
                                line=dict(color='white', width=1)
                            ),
                            text=[f"{event['event']}<br>Sentiment: {event_sentiment:.0f}<br>Price: ${event_price:,.0f}"],
                            hoverinfo='text'
                        ),
                        secondary_y=False
                    )
                    
                    # Add vertical line for the event
                    fig_fg.add_vline(
                        x=event_date,
                        line=dict(
                            color='red' if event['type'] == 'negative' else 'green' if event['type'] == 'positive' else 'gray',
                            width=1,
                            dash='dash'
                        ),
                        opacity=0.3
                    )
            
            # Add halving events markers
            halving_dates = self.fetcher.get_halving_dates()
            for halving_date in halving_dates:
                if halving_date >= fear_greed_df.index[0] and halving_date <= fear_greed_df.index[-1]:
                    # Get sentiment and price at halving
                    halving_sentiment = fear_greed_df[fear_greed_df.index <= halving_date]['value'].iloc[-1]
                    halving_price = price_resampled[price_resampled.index <= halving_date].iloc[-1]
                    
                    # Add marker
                    fig_fg.add_trace(
                        go.Scatter(
                            x=[halving_date],
                            y=[halving_sentiment],
                            mode='markers',
                            name='Bitcoin Halving',
                            marker=dict(
                                size=12,
                                symbol='diamond',
                                color='orange',
                                line=dict(color='white', width=1)
                            ),
                            text=[f"Bitcoin Halving<br>Sentiment: {halving_sentiment:.0f}<br>Price: ${halving_price:,.0f}"],
                            hoverinfo='text'
                        ),
                        secondary_y=False
                    )
                    
                    # Add vertical line for halving
                    fig_fg.add_vline(
                        x=halving_date,
                        line=dict(color='orange', width=1, dash='dash'),
                        opacity=0.3
                    )
            
            # Update layout
            fig_fg.update_layout(
                title='Bitcoin Fear & Greed Index with Price History and Major Events',
                xaxis_title='Date',
                height=700,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                margin=dict(r=70),  # Add right margin for price axis
                hovermode='x unified'
            )
            
            # Update yaxis properties
            fig_fg.update_yaxes(
                title_text="Fear & Greed Index",
                range=[0, 100],
                tickmode='array',
                ticktext=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'],
                tickvals=[12.5, 35, 50, 65, 87.5],
                secondary_y=False
            )
            
            fig_fg.update_yaxes(
                title_text="Bitcoin Price (USD)",
                secondary_y=True,
                showgrid=False
            )
            
            st.plotly_chart(fig_fg, use_container_width=True)
            
            # Event Impact Analysis
            st.subheader("🎯 Event Impact Analysis")
            
            # Create tabs for different analyses
            event_tab1, event_tab2 = st.tabs(["Major Events", "Halving Events"])
            
            with event_tab1:
                # Analyze sentiment changes around major events
                event_impacts = []
                window = 7  # Days to analyze before/after event
                
                for event in events:
                    event_date = event['date']
                    if event_date >= fear_greed_df.index[0] and event_date <= fear_greed_df.index[-1]:
                        pre_event = fear_greed_df[
                            (fear_greed_df.index >= event_date - pd.Timedelta(days=window)) &
                            (fear_greed_df.index < event_date)
                        ]
                        post_event = fear_greed_df[
                            (fear_greed_df.index >= event_date) &
                            (fear_greed_df.index <= event_date + pd.Timedelta(days=window))
                        ]
                        
                        if not pre_event.empty and not post_event.empty:
                            event_impacts.append({
                                'Event': event['event'],
                                'Date': event_date.strftime('%Y-%m-%d'),
                                'Type': event['type'],
                                'Pre-Event Sentiment': pre_event['value'].mean(),
                                'Post-Event Sentiment': post_event['value'].mean(),
                                'Sentiment Change': post_event['value'].mean() - pre_event['value'].mean(),
                                'Pre-Event Label': self.fetcher.get_fear_greed_label(int(pre_event['value'].mean())),
                                'Post-Event Label': self.fetcher.get_fear_greed_label(int(post_event['value'].mean()))
                            })
                
                if event_impacts:
                    impact_df = pd.DataFrame(event_impacts)
                    
                    # Create impact visualization
                    fig_impact = go.Figure()
                    
                    # Add pre-event sentiment bars
                    fig_impact.add_trace(
                        go.Bar(
                            name='Pre-Event',
                            x=impact_df['Event'],
                            y=impact_df['Pre-Event Sentiment'],
                            marker_color='lightblue'
                        )
                    )
                    
                    # Add post-event sentiment bars
                    fig_impact.add_trace(
                        go.Bar(
                            name='Post-Event',
                            x=impact_df['Event'],
                            y=impact_df['Post-Event Sentiment'],
                            marker_color='darkblue'
                        )
                    )
                    
                    fig_impact.update_layout(
                        title=f'Sentiment Impact of Major Events ({window}-Day Window)',
                        barmode='group',
                        xaxis_title='Event',
                        yaxis_title='Sentiment Index',
                        height=500
                    )
                    
                    st.plotly_chart(fig_impact, use_container_width=True)
                    
                    # Show detailed impact table
                    st.dataframe(
                        impact_df.style.format({
                            'Pre-Event Sentiment': '{:.1f}',
                            'Post-Event Sentiment': '{:.1f}',
                            'Sentiment Change': '{:+.1f}'
                        }).background_gradient(
                            subset=['Sentiment Change'],
                            cmap='RdYlGn'
                        ),
                        use_container_width=True
                    )
            
            with event_tab2:
                # Analyze sentiment around halving events
                halving_impacts = []
                halving_window = 30  # Longer window for halving analysis
                
                for halving_date in halving_dates:
                    if halving_date >= fear_greed_df.index[0] and halving_date <= fear_greed_df.index[-1]:
                        pre_halving = fear_greed_df[
                            (fear_greed_df.index >= halving_date - pd.Timedelta(days=halving_window)) &
                            (fear_greed_df.index < halving_date)
                        ]
                        post_halving = fear_greed_df[
                            (fear_greed_df.index >= halving_date) &
                            (fear_greed_df.index <= halving_date + pd.Timedelta(days=halving_window))
                        ]
                        
                        if not pre_halving.empty and not post_halving.empty:
                            halving_impacts.append({
                                'Halving': halving_date.strftime('%Y-%m-%d'),
                                'Pre-Halving Sentiment': pre_halving['value'].mean(),
                                'Post-Halving Sentiment': post_halving['value'].mean(),
                                'Sentiment Change': post_halving['value'].mean() - pre_halving['value'].mean(),
                                'Pre-Halving Label': self.fetcher.get_fear_greed_label(int(pre_halving['value'].mean())),
                                'Post-Halving Label': self.fetcher.get_fear_greed_label(int(post_halving['value'].mean())),
                                'Volatility Change': post_halving['value'].std() - pre_halving['value'].std()
                            })
                
                if halving_impacts:
                    halving_df = pd.DataFrame(halving_impacts)
                    
                    # Create halving impact visualization
                    fig_halving = go.Figure()
                    
                    # Add pre-halving sentiment line
                    fig_halving.add_trace(
                        go.Scatter(
                            x=halving_df['Halving'],
                            y=halving_df['Pre-Halving Sentiment'],
                            name='Pre-Halving',
                            mode='lines+markers',
                            line=dict(color='orange', dash='dash')
                        )
                    )
                    
                    # Add post-halving sentiment line
                    fig_halving.add_trace(
                        go.Scatter(
                            x=halving_df['Halving'],
                            y=halving_df['Post-Halving Sentiment'],
                            name='Post-Halving',
                            mode='lines+markers',
                            line=dict(color='purple')
                        )
                    )
                    
                    fig_halving.update_layout(
                        title=f'Sentiment Around Halving Events ({halving_window}-Day Window)',
                        xaxis_title='Halving Date',
                        yaxis_title='Average Sentiment Index',
                        height=500
                    )
                    
                    st.plotly_chart(fig_halving, use_container_width=True)
                    
                    # Show detailed halving impact table
                    st.dataframe(
                        halving_df.style.format({
                            'Pre-Halving Sentiment': '{:.1f}',
                            'Post-Halving Sentiment': '{:.1f}',
                            'Sentiment Change': '{:+.1f}',
                            'Volatility Change': '{:+.2f}'
                        }).background_gradient(
                            subset=['Sentiment Change'],
                            cmap='RdYlGn'
                        ),
                        use_container_width=True
                    )
            
            # Sentiment Distribution Analysis
            st.subheader("📊 Sentiment Distribution")
            
            # Calculate time spent in each sentiment zone
            sentiment_distribution = pd.cut(
                fear_greed_df['value'],
                bins=[0, 10, 25, 45, 55, 75, 90, 100],
                labels=['Maximum Fear', 'Extreme Fear', 'Fear', 'Neutral', 'Optimism', 'Greed', 'Extreme Greed']
            ).value_counts(normalize=True)
            
            # Convert to percentages and round
            sentiment_distribution = (sentiment_distribution * 100).round(1)
            
            # Create distribution chart
            fig_dist = go.Figure(data=[
                go.Bar(
                    x=sentiment_distribution.index.astype(str),
                    y=sentiment_distribution.values,
                    marker_color=['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen'],
                    text=[f"{val:.1f}%" for val in sentiment_distribution.values],
                    textposition='auto'
                )
            ])
            
            fig_dist.update_layout(
                title='Time Spent in Each Sentiment Zone',
                xaxis_title='Sentiment Level',
                yaxis_title='Percentage of Time (%)',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Price Correlation Analysis
            st.subheader("🔄 Price Correlation Analysis")
            
            # Calculate correlation with price
            price_resampled = prices_df.resample('D')['price'].last()
            merged_data = pd.merge(
                fear_greed_df,
                price_resampled,
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if not merged_data.empty:
                correlation = merged_data['value'].corr(merged_data['price'])
                
                # Create scatter plot
                fig_scatter = px.scatter(
                    merged_data,
                    x='value',
                    y='price',
                    trendline="ols",
                    title='Fear & Greed Index vs. Bitcoin Price',
                    labels={'value': 'Fear & Greed Index', 'price': 'Bitcoin Price (USD)'}
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Price Correlation", f"{correlation:.2f}")
                
                # Trading Analysis
                st.subheader("💡 Trading Insights")
                
                # Show average returns based on sentiment
                returns = merged_data['price'].pct_change().dropna()
                if len(returns) > 0:
                    sentiment_values = merged_data['value'].shift(1)
                    valid_mask = ~sentiment_values.isna()
                    
                    sentiment_returns = pd.DataFrame({
                        'Sentiment': pd.cut(
                            sentiment_values[valid_mask],
                            bins=[0, 25, 75, 100],
                            labels=['Fear', 'Neutral', 'Greed']
                        ),
                        'Return': returns[valid_mask]
                    })
                    
                    if not sentiment_returns.empty and not sentiment_returns['Sentiment'].isna().all():
                        avg_returns = sentiment_returns.groupby('Sentiment', observed=True)['Return'].agg([
                            'mean', 'std', 'count'
                        ])
                        avg_returns['mean'] *= 100  # Convert to percentage
                        avg_returns['std'] *= 100
                        
                        # Display returns table
                        st.write("**Average Next-Day Returns by Sentiment:**")
                        st.dataframe(
                            avg_returns.style.format({
                                'mean': format_metrics,
                                'std': format_metrics,
                                'count': '{:.0f} days'
                            }),
                            use_container_width=True
                        )
                        
                        # Add trading suggestions
                        st.write("\n**Current Trading Signals:**")
                        if current_fg <= 25:
                            st.success("🎯 Current extreme fear levels often present buying opportunities")
                            st.write(f"Historical average return after fear: {format_metrics(avg_returns.loc['Fear', 'mean'])}")
                        elif current_fg >= 75:
                            st.warning("⚠️ Current high greed levels suggest caution")
                            st.write(f"Historical average return after greed: {format_metrics(avg_returns.loc['Greed', 'mean'])}")
                        else:
                            st.info("💫 Market sentiment is neutral")
                            st.write(f"Historical average return in neutral periods: {format_metrics(avg_returns.loc['Neutral', 'mean'])}")
                    else:
                        st.info("Insufficient data to calculate sentiment-based returns.")
                else:
                    st.info("Insufficient price data to calculate returns.")
            else:
                st.info("No overlapping data between price and sentiment indices.")
        
        except Exception as e:
            st.error(f"Error fetching sentiment data: {str(e)}")
            st.info("Please try again later or contact support if the problem persists.")

    def show_onchain_analysis(self, prices_df: pd.DataFrame):
        """Show on-chain analysis page."""
        st.title("On-chain Analysis")
        
        try:
            # Get date range from prices_df
            start_date = prices_df.index[0].strftime('%Y-%m-%d')
            end_date = prices_df.index[-1].strftime('%Y-%m-%d')
            
            # Fetch on-chain data
            with st.spinner("Fetching on-chain data..."):
                onchain_data = self.fetcher.fetch_onchain_data(
                    start_date=start_date,
                    end_date=end_date
                )
            
            if onchain_data.empty:
                st.warning("No on-chain data available for the selected date range.")
                return
                
            # Analyze metrics
            analysis_results = self.fetcher.analyze_onchain_metrics(
                prices_df,
                onchain_data
            )
            
            # Market Overview
            st.subheader("🌍 Market Overview")
            col1, col2, col3 = st.columns(3)
            
            # Get current metrics
            current_metrics = analysis_results.get("current", {})
            with col1:
                st.metric(
                    "Market Cap",
                    f"${current_metrics.get('market_cap', 0):,.0f}B",
                    f"{current_metrics.get('market_cap_change', 0):.1f}%"
                )
            with col2:
                st.metric(
                    "Realized Cap",
                    f"${current_metrics.get('realized_cap', 0):,.0f}B",
                    f"{current_metrics.get('realized_cap_change', 0):.1f}%"
                )
            with col3:
                st.metric(
                    "MVRV Ratio",
                    f"{current_metrics.get('mvrv', 1.0):.2f}",
                    f"{current_metrics.get('mvrv_change', 0):.1f}%"
                )
            
            # Create tabs for different analyses
            tabs = st.tabs([
                "📊 Network Activity",
                "💎 HODL Analysis",
                "⛏️ Mining Metrics",
                "💱 Exchange Flows",
                "📈 Market Valuation"
            ])
            
            with tabs[0]:
                self._plot_network_metrics(analysis_results.get("network", {}))
                
                # Add network health score
                health_score = analysis_results.get("network", {}).get("health_score", 0)
                st.info(f"🏥 Network Health Score: {health_score:.0f}/100")
                
                # Add key insights
                insights = analysis_results.get("network", {}).get("insights", [])
                if insights:
                    st.subheader("🔍 Key Insights")
                    for insight in insights:
                        st.write(f"• {insight}")
            
            with tabs[1]:
                self._plot_hodl_waves(analysis_results.get("hodl", {}))
                
                # Add HODL analysis
                st.subheader("💎 HODL Analysis")
                hodl_stats = analysis_results.get("hodl", {}).get("statistics", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Diamond Hands Index",
                        f"{hodl_stats.get('diamond_hands_index', 0):.1f}",
                        f"{hodl_stats.get('diamond_hands_change', 0):.1f}%"
                    )
                with col2:
                    st.metric(
                        "Average HODL Time",
                        f"{hodl_stats.get('avg_hodl_time', 0):.1f} days",
                        f"{hodl_stats.get('hodl_time_change', 0):.1f}%"
                    )
            
            with tabs[2]:
                self._plot_mining_metrics(analysis_results.get("mining", {}))
                
                # Add mining profitability analysis
                st.subheader("⛏️ Mining Profitability")
                mining_stats = analysis_results.get("mining", {}).get("profitability", {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Break-even Price",
                        f"${mining_stats.get('breakeven_price', 0):,.0f}",
                        f"{mining_stats.get('breakeven_change', 0):.1f}%"
                    )
                with col2:
                    st.metric(
                        "Miner Revenue (USD/TH/day)",
                        f"${mining_stats.get('revenue_per_th', 0):.2f}",
                        f"{mining_stats.get('revenue_change', 0):.1f}%"
                    )
                with col3:
                    st.metric(
                        "Network Security",
                        f"{mining_stats.get('security_score', 0)}/100",
                        f"{mining_stats.get('security_change', 0):.1f}%"
                    )
            
            with tabs[3]:
                self._plot_exchange_flows(analysis_results.get("flows", {}))
                
                # Add exchange analysis
                st.subheader("💱 Exchange Analysis")
                exchange_stats = analysis_results.get("flows", {}).get("statistics", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Exchange Supply Ratio",
                        f"{exchange_stats.get('supply_ratio', 0):.1f}%",
                        f"{exchange_stats.get('supply_ratio_change', 0):.1f}%"
                    )
                with col2:
                    st.metric(
                        "Accumulation Score",
                        f"{exchange_stats.get('accumulation_score', 0)}/100",
                        f"{exchange_stats.get('accumulation_change', 0):.1f}%"
                    )
            
            with tabs[4]:
                # Market Valuation Metrics
                st.subheader("📈 Market Valuation Metrics")
                
                valuation = analysis_results.get("valuation", {})
                metrics = [
                    ("MVRV Z-Score", "mvrv_zscore", "Measures if Bitcoin is over/undervalued relative to its fair value"),
                    ("Stock-to-Flow Ratio", "stock_to_flow", "Measures scarcity based on current supply and production rate"),
                    ("Puell Multiple", "puell_multiple", "Indicates if mining revenue is high or low relative to yearly average"),
                    ("Net Unrealized Profit/Loss", "nupl", "Shows the overall market's unrealized profit/loss as a proportion of market cap")
                ]
                
                for name, key, description in metrics:
                    value = valuation.get(key, 0)
                    change = valuation.get(f"{key}_change", 0)
                    signal = valuation.get(f"{key}_signal", "Neutral")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.metric(
                            name,
                            f"{value:.2f}",
                            f"{change:+.1f}%"
                        )
                        st.caption(description)
                    with col2:
                        signal_color = {
                            "Bullish": "success",
                            "Bearish": "error",
                            "Neutral": "info"
                        }.get(signal, "info")
                        st.markdown(f":{signal_color}[Signal: {signal}]")
                
                # Market Cycle Analysis
                st.subheader("🔄 Market Cycle Analysis")
                cycle_metrics = analysis_results.get("cycle", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Cycle Phase",
                        cycle_metrics.get("current_phase", "Unknown"),
                        cycle_metrics.get("phase_duration", "0") + " days"
                    )
                with col2:
                    st.metric(
                        "Cycle Progress",
                        f"{cycle_metrics.get('cycle_progress', 0):.0f}%",
                        cycle_metrics.get("cycle_prediction", "Unknown")
                    )
            
            # Overall Market Analysis
            with st.expander("📊 Overall Market Analysis"):
                st.write(analysis_results.get("summary", "No market analysis available."))
                
                # Risk Metrics
                st.subheader("🎯 Risk Metrics")
                risk_metrics = analysis_results.get("risk", {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Market Risk Score",
                        f"{risk_metrics.get('market_risk', 50)}/100",
                        risk_metrics.get("risk_signal", "Neutral")
                    )
                with col2:
                    st.metric(
                        "Volatility Risk",
                        f"{risk_metrics.get('volatility_risk', 50)}/100",
                        f"{risk_metrics.get('volatility_change', 0):+.1f}%"
                    )
                with col3:
                    st.metric(
                        "Liquidity Risk",
                        f"{risk_metrics.get('liquidity_risk', 50)}/100",
                        f"{risk_metrics.get('liquidity_change', 0):+.1f}%"
                    )
            
        except Exception as e:
            st.error(f"Error analyzing on-chain data: {str(e)}")
            st.info("Please try again later or contact support if the problem persists.")

    def run(self):
        """Run the Streamlit application."""
        st.sidebar.title("🎯 Bitcoin Festival Tracker")
        
        # Main view selector at the top
        view_type = st.sidebar.radio(
            "📊 Select View",
            ["Historical Analysis", "Future Predictions"],
            help="Choose between analyzing past performance or viewing future predictions"
        )
        
        # Initialize date variables
        now = pd.Timestamp.now()
        max_future_date = pd.Timestamp('2030-12-31')
        
        if view_type == "Historical Analysis":
            st.sidebar.subheader("📅 Historical Range")
            time_range = st.sidebar.selectbox(
                "Select Time Period",
                [
                    "Last 6 Months",
                    "Last Year",
                    "Last 3 Years",
                    "Last 5 Years",
                    "All Historical Data",
                    "Custom Range"
                ]
            )
            
            # Map selection to date range
            if time_range == "Last 6 Months":
                start_date = now - pd.Timedelta(days=180)
                end_date = now
            elif time_range == "Last Year":
                start_date = now - pd.Timedelta(days=365)
                end_date = now
            elif time_range == "Last 3 Years":
                start_date = now - pd.Timedelta(days=3*365)
                end_date = now
            elif time_range == "Last 5 Years":
                start_date = now - pd.Timedelta(days=5*365)
                end_date = now
            elif time_range == "All Historical Data":
                start_date = pd.Timestamp('2014-01-01')
                end_date = now
            else:  # Custom Range
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    start_date = pd.Timestamp(st.date_input(
                        "Start Date",
                        value=now - pd.Timedelta(days=365),
                        min_value=pd.Timestamp('2014-01-01').date(),
                        max_value=now.date()
                    ))
                with col2:
                    end_date = pd.Timestamp(st.date_input(
                        "End Date",
                        value=now.date(),
                        min_value=start_date.date(),
                        max_value=now.date()
                    ))
            
            # Navigation for historical analysis
            page = st.sidebar.radio(
                "📱 Navigation",
                ["Overview", "Festival Analysis", "Bull & Bear", "Market Sentiment", "On-chain Analysis", "Drawdown & Volatility"],
                help="Choose what analysis to view"
            )
            
        else:  # Future Predictions
            st.sidebar.subheader("🔮 Prediction Range")
            prediction_range = st.sidebar.selectbox(
                "Select Prediction Horizon",
                [
                    "Next 6 Months",
                    "Next Year",
                    "Next 2 Years",
                    "All Future Festivals",
                    "Custom Range"
                ]
            )
            
            # Always include 1 year of historical data for context
            start_date = now - pd.Timedelta(days=365)
            
            # Map selection to end date
            if prediction_range == "Next 6 Months":
                end_date = now + pd.Timedelta(days=180)
            elif prediction_range == "Next Year":
                end_date = now + pd.Timedelta(days=365)
            elif prediction_range == "Next 2 Years":
                end_date = now + pd.Timedelta(days=2*365)
            elif prediction_range == "All Future Festivals":
                end_date = max_future_date
            else:  # Custom Range
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    custom_start = pd.Timestamp(st.date_input(
                        "Start Date",
                        value=now.date(),
                        min_value=now.date(),
                        max_value=max_future_date.date()
                    ))
                with col2:
                    end_date = pd.Timestamp(st.date_input(
                        "End Date",
                        value=(now + pd.Timedelta(days=180)).date(),
                        min_value=custom_start.date(),
                        max_value=max_future_date.date()
                    ))
            
            # Force Upcoming Festivals view for predictions
            page = "Upcoming Festivals"
            st.sidebar.info(f"📅 Showing festivals from {now.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get historical price data (only up to current date)
        prices_df = self.db.get_bitcoin_prices(
            start_date.strftime('%Y-%m-%d'),
            min(now, end_date).strftime('%Y-%m-%d')
        )
        
        if prices_df.empty:
            st.error("No price data available for the selected date range.")
            return
        
        # Get festival data based on view type
        festivals = self.db.get_festivals()
        if view_type == "Historical Analysis":
            festivals = festivals[
                (pd.to_datetime(festivals['end_date']) <= now) &
                (pd.to_datetime(festivals['start_date']) >= start_date)
            ]
        else:
            festivals = festivals[
                (pd.to_datetime(festivals['start_date']) >= now) &
                (pd.to_datetime(festivals['start_date']) <= end_date)
            ]
        
        # Calculate statistics
        stats_df = self.fetcher.analyze_festival_performance(prices_df, festivals)
        
        # Show appropriate view
        if page == "Overview":
            self.show_overview(prices_df, stats_df)
        elif page == "Festival Analysis":
            self.show_festival_analysis(prices_df, stats_df)
        elif page == "Bull & Bear":
            self.show_bull_bear_analysis(prices_df)
        elif page == "Market Sentiment":
            self.show_sentiment_analysis(prices_df)
        elif page == "On-chain Analysis":
            self.show_onchain_analysis(prices_df)
        elif page == "Drawdown & Volatility":
            self.show_drawdown_analysis(prices_df, stats_df, time_range)
        else:  # Upcoming Festivals
            self.show_upcoming_festivals(prices_df, end_date)

    def _plot_network_metrics(self, network_data: Dict):
        """Plot network activity metrics."""
        if not network_data:
            st.warning("No network activity data available.")
            return
            
        # Create metrics overview
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                active_addresses = network_data.get('active_addresses', 0)
                active_addresses = float(active_addresses) if isinstance(active_addresses, (int, float, str)) else active_addresses.iloc[-1]
                active_change = network_data.get('active_addresses_change', 0)
                active_change = float(active_change) if isinstance(active_change, (int, float, str)) else active_change.iloc[-1]
                st.metric(
                    "Active Addresses",
                    f"{int(active_addresses):,}",
                    f"{active_change:.1f}%"
                )
            except Exception:
                st.metric("Active Addresses", "N/A", "N/A")

        with col2:
            try:
                tx_count = network_data.get('transaction_count', 0)
                tx_count = float(tx_count) if isinstance(tx_count, (int, float, str)) else tx_count.iloc[-1]
                tx_change = network_data.get('transaction_count_change', 0)
                tx_change = float(tx_change) if isinstance(tx_change, (int, float, str)) else tx_change.iloc[-1]
                st.metric(
                    "Transaction Count",
                    f"{int(tx_count):,}",
                    f"{tx_change:.1f}%"
                )
            except Exception:
                st.metric("Transaction Count", "N/A", "N/A")

        with col3:
            try:
                hash_rate = network_data.get('hash_rate', 0)
                hash_rate = float(hash_rate) if isinstance(hash_rate, (int, float, str)) else hash_rate.iloc[-1]
                hash_change = network_data.get('hash_rate_change', 0)
                hash_change = float(hash_change) if isinstance(hash_change, (int, float, str)) else hash_change.iloc[-1]
                st.metric(
                    "Network Hash Rate",
                    f"{hash_rate:,.0f} TH/s",
                    f"{hash_change:.1f}%"
                )
            except Exception:
                st.metric("Network Hash Rate", "N/A", "N/A")
            
        # Plot time series
        if all(k in network_data for k in ['dates', 'prices', 'active_addresses_trend']):
            try:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add price line
                fig.add_trace(
                    go.Scatter(
                        x=network_data['dates'],
                        y=network_data['prices'],
                        name='Bitcoin Price',
                        line=dict(color='gray', width=1)
                    ),
                    secondary_y=True
                )
                
                # Add network metrics
                fig.add_trace(
                    go.Scatter(
                        x=network_data['dates'],
                        y=network_data['active_addresses_trend'],
                        name='Active Addresses',
                        line=dict(color='blue')
                    ),
                    secondary_y=False
                )
                
                fig.update_layout(
                    title='Network Activity vs Price',
                    xaxis_title='Date',
                    yaxis_title='Active Addresses',
                    yaxis2_title='Price (USD)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting network metrics: {str(e)}")
        
    def _plot_hodl_waves(self, hodl_data: Dict):
        """Plot HODL wave analysis."""
        if not hodl_data:
            st.warning("No HODL wave data available.")
            return
            
        # Create metrics overview
        col1, col2 = st.columns(2)
        with col1:
            try:
                lth_supply = hodl_data.get('lth_supply', 0)
                lth_supply = float(lth_supply) if isinstance(lth_supply, (int, float, str)) else lth_supply.iloc[-1]
                lth_change = hodl_data.get('lth_supply_change', 0)
                lth_change = float(lth_change) if isinstance(lth_change, (int, float, str)) else lth_change.iloc[-1]
                st.metric(
                    "Long-term Holder Supply",
                    f"{lth_supply:.1f}%",
                    f"{lth_change:.1f}%"
                )
            except Exception:
                st.metric("Long-term Holder Supply", "N/A", "N/A")

        with col2:
            try:
                sth_supply = hodl_data.get('sth_supply', 0)
                sth_supply = float(sth_supply) if isinstance(sth_supply, (int, float, str)) else sth_supply.iloc[-1]
                sth_change = hodl_data.get('sth_supply_change', 0)
                sth_change = float(sth_change) if isinstance(sth_change, (int, float, str)) else sth_change.iloc[-1]
                st.metric(
                    "Short-term Holder Supply",
                    f"{sth_supply:.1f}%",
                    f"{sth_change:.1f}%"
                )
            except Exception:
                st.metric("Short-term Holder Supply", "N/A", "N/A")
            
        # Plot HODL waves
        if 'waves' in hodl_data and 'dates' in hodl_data:
            try:
                fig = go.Figure()
                
                for age_band, values in hodl_data['waves'].items():
                    fig.add_trace(
                        go.Scatter(
                            x=hodl_data['dates'],
                            y=values,
                            name=age_band,
                            stackgroup='one'
                        )
                    )
                    
                fig.update_layout(
                    title='HODL Waves',
                    xaxis_title='Date',
                    yaxis_title='% of Supply',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting HODL waves: {str(e)}")
        
    def _plot_mining_metrics(self, mining_data: Dict):
        """Plot mining-related metrics."""
        if not mining_data:
            st.warning("No mining data available.")
            return
            
        # Create metrics overview
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                hash_rate = mining_data.get('hash_rate', 0)
                hash_rate = float(hash_rate) if isinstance(hash_rate, (int, float, str)) else hash_rate.iloc[-1]
                hash_change = mining_data.get('hash_rate_change', 0)
                hash_change = float(hash_change) if isinstance(hash_change, (int, float, str)) else hash_change.iloc[-1]
                st.metric(
                    "Hash Rate",
                    f"{hash_rate:,.0f} TH/s",
                    f"{hash_change:.1f}%"
                )
            except Exception:
                st.metric("Hash Rate", "N/A", "N/A")

        with col2:
            try:
                revenue = mining_data.get('revenue', 0)
                revenue = float(revenue) if isinstance(revenue, (int, float, str)) else revenue.iloc[-1]
                rev_change = mining_data.get('revenue_change', 0)
                rev_change = float(rev_change) if isinstance(rev_change, (int, float, str)) else rev_change.iloc[-1]
                st.metric(
                    "Mining Revenue",
                    f"${revenue:,.0f}",
                    f"{rev_change:.1f}%"
                )
            except Exception:
                st.metric("Mining Revenue", "N/A", "N/A")

        with col3:
            try:
                difficulty = mining_data.get('difficulty', 0)
                difficulty = float(difficulty) if isinstance(difficulty, (int, float, str)) else difficulty.iloc[-1]
                diff_change = mining_data.get('difficulty_change', 0)
                diff_change = float(diff_change) if isinstance(diff_change, (int, float, str)) else diff_change.iloc[-1]
                st.metric(
                    "Difficulty",
                    f"{difficulty:,.0f}",
                    f"{diff_change:.1f}%"
                )
            except Exception:
                st.metric("Difficulty", "N/A", "N/A")
            
        # Plot mining metrics
        if all(k in mining_data for k in ['dates', 'hash_rate_trend', 'revenue_trend']):
            try:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(
                        x=mining_data['dates'],
                        y=mining_data['hash_rate_trend'],
                        name='Hash Rate',
                        line=dict(color='orange')
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=mining_data['dates'],
                        y=mining_data['revenue_trend'],
                        name='Mining Revenue',
                        line=dict(color='green')
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title='Mining Metrics Over Time',
                    xaxis_title='Date',
                    yaxis_title='Hash Rate (TH/s)',
                    yaxis2_title='Revenue (USD)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting mining metrics: {str(e)}")
        
    def _plot_exchange_flows(self, flow_data: Dict):
        """Plot exchange flow metrics."""
        if not flow_data:
            st.warning("No exchange flow data available.")
            return
            
        # Create metrics overview
        col1, col2 = st.columns(2)
        with col1:
            try:
                net_flow = flow_data.get('net_flow', 0)
                net_flow = float(net_flow) if isinstance(net_flow, (int, float, str)) else net_flow.iloc[-1]
                net_change = flow_data.get('net_flow_change', 0)
                net_change = float(net_change) if isinstance(net_change, (int, float, str)) else net_change.iloc[-1]
                st.metric(
                    "Net Flow",
                    f"{net_flow:,.0f} BTC",
                    f"{net_change:.1f}%"
                )
            except Exception:
                st.metric("Net Flow", "N/A", "N/A")

        with col2:
            try:
                balance = flow_data.get('exchange_balance', 0)
                balance = float(balance) if isinstance(balance, (int, float, str)) else balance.iloc[-1]
                balance_change = flow_data.get('exchange_balance_change', 0)
                balance_change = float(balance_change) if isinstance(balance_change, (int, float, str)) else balance_change.iloc[-1]
                st.metric(
                    "Exchange Balance",
                    f"{balance:,.0f} BTC",
                    f"{balance_change:.1f}%"
                )
            except Exception:
                st.metric("Exchange Balance", "N/A", "N/A")
            
        # Plot exchange flows
        if all(k in flow_data for k in ['dates', 'inflow', 'outflow']):
            try:
                fig = go.Figure()
                
                fig.add_trace(
                    go.Bar(
                        x=flow_data['dates'],
                        y=flow_data['inflow'],
                        name='Inflow',
                        marker_color='red'
                    )
                )
                
                fig.add_trace(
                    go.Bar(
                        x=flow_data['dates'],
                        y=flow_data['outflow'],
                        name='Outflow',
                        marker_color='green'
                    )
                )
                
                fig.update_layout(
                    title='Exchange Flows',
                    xaxis_title='Date',
                    yaxis_title='BTC',
                    height=500,
                    barmode='relative'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting exchange flows: {str(e)}")

def main():
    """Main entry point for the application."""
    app = BitcoinFestivalApp()
    app.run()

if __name__ == "__main__":
    main() 