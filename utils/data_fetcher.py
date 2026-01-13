import yfinance as yf
import pandas as pd
import numpy as np
import random
import time
import requests
import os
from datetime import datetime, timedelta
import streamlit as st
from utils.request_throttler import get_throttler

# Load API keys from environment variables or config
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')

# User agents for API request rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
]

def get_available_markets():
    """Return a list of available stock markets."""
    return ["Global", "NSE (India)", "BSE (India)"]


def _get_finnhub_data(symbol, start_date, end_date):
    """
    Fetch data from Finnhub API (Primary - best rate limits: ~60/min)
    """
    if not FINNHUB_API_KEY:
        raise Exception("Finnhub API key not configured")
    
    try:
        # Finnhub doesn't provide historical data in free tier, so we use it for validation
        # and recent quote data, then fall back to other APIs for historical data
        url = f"https://finnhub.io/api/v1/quote"
        params = {"symbol": symbol, "token": FINNHUB_API_KEY}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'c' not in data or data['c'] is None:
            raise Exception(f"No data from Finnhub for {symbol}")
        
        # Finnhub provides current quote but not historical data in free tier
        # So we validate the symbol works, then use yfinance for actual historical data
        return None  # Will fall through to next API
        
    except requests.exceptions.Timeout:
        raise Exception("Finnhub request timeout")
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            raise Exception(f"Finnhub rate limit: {str(e)}")
        raise Exception(f"Finnhub error: {str(e)}")


def _get_alpha_vantage_data(symbol, start_date, end_date):
    """
    Fetch data from Alpha Vantage API (Secondary - 5 calls/min, 500/day)
    """
    if not ALPHA_VANTAGE_API_KEY:
        raise Exception("Alpha Vantage API key not configured")
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "full"
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            error_msg = data.get("Note", data.get("Error Message", "No data found"))
            raise Exception(f"Alpha Vantage: {error_msg}")
        
        # Convert to DataFrame
        ts_data = data["Time Series (Daily)"]
        df_data = []
        
        for date_str, values in ts_data.items():
            date = pd.to_datetime(date_str)
            if date < start_date or date > end_date:
                continue
                
            df_data.append({
                'Date': date,
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': float(values['5. volume']),
                'Adj Close': float(values['4. close'])
            })
        
        if not df_data:
            raise Exception(f"No data for {symbol} in date range")
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Date').set_index('Date')
        return df
        
    except requests.exceptions.Timeout:
        raise Exception("Alpha Vantage request timeout")
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower() or "call frequency" in str(e).lower():
            raise Exception(f"Alpha Vantage rate limit: {str(e)}")
        raise Exception(f"Alpha Vantage error: {str(e)}")


def _get_yfinance_data(symbol, start_date, end_date):
    """
    Fetch data from yfinance (Fallback - free but has rate limits)
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date + timedelta(days=1))
        
        if data.empty:
            raise Exception(f"No data found for {symbol}")
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise Exception(f"Invalid data structure for {symbol}")
        
        return data
        
    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str or "too many" in error_str or "429" in error_str:
            raise Exception(f"yfinance rate limit: {str(e)}")
        raise Exception(f"yfinance error: {str(e)}")


@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(symbol, start_date, end_date):
    """
    Fetches stock data with intelligent multi-API fallback strategy.
    
    Primary APIs (in order of attempt):
    1. Alpha Vantage (5 calls/min, 500/day) - Good data quality
    2. yfinance (fallback) - Free but has rate limits
    
    Finnhub can validate symbols but doesn't provide historical data in free tier.
    
    Args:
        symbol (str): Stock symbol (e.g., AAPL, RELIANCE.NS)
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
        
    Returns:
        pandas.DataFrame: Historical stock data
        
    Raises:
        Exception: If all APIs fail
    """
    import yfinance as yf
    
    # Validate inputs
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Stock symbol must be a non-empty string")
    
    symbol = symbol.strip().upper()
    throttler = get_throttler()
    
    # Check if we should wait before requesting
    wait_time = throttler.wait_if_needed(symbol)
    if wait_time:
        st.info(f"⏳ Waiting {wait_time:.0f}s before retry (respecting API limits)...")
        time.sleep(wait_time)
    
    # Try multiple APIs in order
    api_errors = {}
    
    # API 1: Try Alpha Vantage first (if configured)
    if ALPHA_VANTAGE_API_KEY:
        try:
            st.info("📡 Fetching from Alpha Vantage...")
            data = _get_alpha_vantage_data(symbol, start_date, end_date)
            throttler.record_request(symbol)
            st.success("✅ Data fetched from Alpha Vantage")
            return data
        except Exception as e:
            api_errors['Alpha Vantage'] = str(e)
            # Check if rate limited
            if "rate limit" in str(e).lower():
                throttler.record_rate_limit(symbol)
    
    # API 2: Try yfinance (always available)
    try:
        st.info("📡 Fetching from yfinance...")
        data = _get_yfinance_data(symbol, start_date, end_date)
        throttler.record_request(symbol)
        st.success("✅ Data fetched from yfinance")
        return data
    except Exception as e:
        api_errors['yfinance'] = str(e)
        error_str = str(e).lower()
        if "rate limit" in error_str or "too many" in error_str:
            throttler.record_rate_limit(symbol)
    
    # All APIs failed - provide helpful error message
    is_rate_limit = any(
        "rate limit" in api_errors[api].lower() 
        for api in api_errors
    )
    
    is_symbol_error = any(
        "not found" in api_errors[api].lower() or "no data" in api_errors[api].lower()
        for api in api_errors
    )
    
    if is_rate_limit:
        raise Exception(
            f"🔄 **Multiple APIs Rate Limited**\n\n"
            f"All stock APIs are temporarily rate-limited due to high traffic.\n\n"
            f"**What to do:**\n"
            f"1. **Try a different stock** - Try AAPL, MSFT, or GOOGL\n"
            f"2. **Check back in 2-3 minutes** - All limits reset quickly\n"
            f"3. **Use your cached data** - Refresh to see previously loaded stocks\n\n"
            f"**Why this happens:**\n"
            f"Multiple free APIs are shared by thousands of users. "
            f"During busy times (like market open), everyone hits limits simultaneously.\n\n"
            f"**After waiting:**\n"
            f"Come back and try {symbol} again - it should work!"
        )
    elif is_symbol_error:
        raise Exception(
            f"**❌ Stock Not Found**\n\n"
            f"'{symbol}' doesn't exist or has no data.\n\n"
            f"**Try these instead:**\n\n"
            f"**US Stocks:**\n"
            f"• AAPL - Apple\n"
            f"• MSFT - Microsoft\n"
            f"• GOOGL - Google\n"
            f"• TSLA - Tesla\n\n"
            f"**Indian Stocks (NSE):**\n"
            f"• RELIANCE.NS - Reliance\n"
            f"• TCS.NS - Tata Consultancy\n"
            f"• INFY.NS - Infosys\n"
            f"• HDFCBANK.NS - HDFC Bank"
        )
    else:
        # Network or other error
        raise Exception(
            f"**Unable to fetch {symbol}**\n\n"
            f"**Errors from all APIs:**\n"
            + "\n".join([f"• {api}: {api_errors[api]}" for api in api_errors]) +
            f"\n\n**Try:**\n"
            f"• Check your internet connection\n"
            f"• Wait a moment and try again\n"
            f"• Try a different stock symbol"
        )

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_stock_info(symbol):
    """
    Get detailed information about a stock.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Stock information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info
    except Exception as e:
        raise Exception(f"Error fetching stock information: {str(e)}")

def format_indian_stock_symbol(symbol, exchange):
    """
    Formats stock symbols for Indian exchanges if not already formatted.
    
    Args:
        symbol (str): Stock symbol
        exchange (str): 'NSE' or 'BSE'
        
    Returns:
        str: Properly formatted stock symbol
    """
    if exchange == "NSE" and not symbol.endswith(".NS"):
        return f"{symbol}.NS"
    elif exchange == "BSE" and not symbol.endswith(".BO"):
        return f"{symbol}.BO"
    return symbol

def validate_stock_symbol(symbol):
    """
    Validates if a stock symbol exists.
    
    Args:
        symbol (str): Stock symbol to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # If we can fetch market cap, the stock likely exists
        if 'marketCap' in info and info['marketCap'] is not None:
            return True
        return False
    except:
        return False


def get_income_statement(symbol):
    """
    Fetch income statement data from Alpha Vantage.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Annual and quarterly income statements
    """
    if not ALPHA_VANTAGE_API_KEY:
        st.warning("Alpha Vantage API key not configured. Income statements unavailable.")
        return None
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        time.sleep(1.2)  # Rate limiting
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'annualReports' in data:
            return {
                'annual': data.get('annualReports', []),
                'quarterly': data.get('quarterlyReports', [])
            }
        else:
            return None
            
    except Exception as e:
        st.warning(f"Could not fetch income statement: {str(e)}")
        return None


def get_balance_sheet(symbol):
    """
    Fetch balance sheet data from Alpha Vantage.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Annual and quarterly balance sheets
    """
    if not ALPHA_VANTAGE_API_KEY:
        st.warning("Alpha Vantage API key not configured. Balance sheets unavailable.")
        return None
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "BALANCE_SHEET",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        time.sleep(1.2)  # Rate limiting
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'annualReports' in data:
            return {
                'annual': data.get('annualReports', []),
                'quarterly': data.get('quarterlyReports', [])
            }
        else:
            return None
            
    except Exception as e:
        st.warning(f"Could not fetch balance sheet: {str(e)}")
        return None


def get_cash_flow(symbol):
    """
    Fetch cash flow statement data from Alpha Vantage.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Annual and quarterly cash flow statements
    """
    if not ALPHA_VANTAGE_API_KEY:
        st.warning("Alpha Vantage API key not configured. Cash flow unavailable.")
        return None
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "CASH_FLOW",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        time.sleep(1.2)  # Rate limiting
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'annualReports' in data:
            return {
                'annual': data.get('annualReports', []),
                'quarterly': data.get('quarterlyReports', [])
            }
        else:
            return None
            
    except Exception as e:
        st.warning(f"Could not fetch cash flow: {str(e)}")
        return None


def get_company_overview(symbol):
    """
    Fetch comprehensive company overview from Alpha Vantage.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Company information, financial ratios, and key metrics
    """
    if not ALPHA_VANTAGE_API_KEY:
        st.warning("Alpha Vantage API key not configured. Company overview unavailable.")
        return None
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        time.sleep(1.2)  # Rate limiting
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'Name' in data:  # Valid response
            return data
        else:
            return None
            
    except Exception as e:
        st.warning(f"Could not fetch company overview: {str(e)}")
        return None


def get_earnings(symbol):
    """
    Fetch earnings history from Alpha Vantage.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Annual and quarterly earnings data
    """
    if not ALPHA_VANTAGE_API_KEY:
        st.warning("Alpha Vantage API key not configured. Earnings unavailable.")
        return None
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "EARNINGS",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        time.sleep(1.2)  # Rate limiting
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'annualEarnings' in data:
            return {
                'annual': data.get('annualEarnings', []),
                'quarterly': data.get('quarterlyEarnings', [])
            }
        else:
            return None
            
    except Exception as e:
        st.warning(f"Could not fetch earnings: {str(e)}")
        return None
