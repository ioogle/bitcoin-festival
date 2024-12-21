"""Main Streamlit application for Bitcoin Festival Price Tracker."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.absolute())
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

from data.database import Database
from data.fetcher import (
    fetch_bitcoin_prices,
    get_festivals_data,
    analyze_festival_performance
)
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
                prices = fetch_bitcoin_prices()
                if not prices.empty:
                    self.db.store_bitcoin_prices(prices)
                else:
                    st.error("Failed to fetch Bitcoin price data.")
                    st.stop()
        
        # Load festivals if not exists
        festivals = self.db.get_festivals()
        if len(festivals) == 0:
            festivals_df = get_festivals_data()
            self.db.store_festivals(festivals_df)

    def show_overview(self, prices_df: pd.DataFrame, stats_df: pd.DataFrame):
        """Show overview dashboard with key metrics."""
        st.header("📊 Overview")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_festival = stats_df.loc[stats_df['Price Change %'].idxmax()]
            st.metric(
                "Best Performing Festival",
                best_festival['Festival'],
                format_metrics(best_festival['Price Change %'])
            )
        
        with col2:
            avg_return = stats_df['Price Change %'].mean()
            success_rate = (stats_df['Price Change %'] > 0).mean() * 100
            st.metric(
                "Average Festival Return",
                format_metrics(avg_return),
                f"{success_rate:.0f}% Success Rate"
            )
        
        with col3:
            upcoming = self.db.get_upcoming_festivals()
            st.metric(
                "Upcoming Festivals",
                len(upcoming),
                f"Next: {upcoming.iloc[0]['name'] if not upcoming.empty else 'None'}"
            )
        
        # Price chart
        st.plotly_chart(create_price_chart(prices_df), use_container_width=True)
        
        # Festival performance
        st.plotly_chart(create_festival_performance_chart(stats_df), use_container_width=True)

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
                stats = analyze_festival_performance(prices_df, historical)
                
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
                st.subheader("📊 Scenario Analysis")
                
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
                ["Overview", "Festival Analysis"],
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
        stats_df = analyze_festival_performance(prices_df, festivals)
        
        # Show appropriate view
        if page == "Overview":
            self.show_overview(prices_df, stats_df)
        elif page == "Festival Analysis":
            self.show_festival_analysis(prices_df, stats_df)
        else:  # Upcoming Festivals
            self.show_upcoming_festivals(prices_df, end_date)

def main():
    """Main entry point for the application."""
    app = BitcoinFestivalApp()
    app.run()

if __name__ == "__main__":
    main() 