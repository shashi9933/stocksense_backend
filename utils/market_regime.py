"""
Market Regime Detection module for stock analysis.

This module provides functions to detect different market regimes such as trending up,
trending down, or range-bound (sideways) markets. These regime classifications
can be used to select the most appropriate prediction models for current market conditions.
"""

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def calculate_atr(data, window=14):
    """
    Calculate Average True Range (ATR) for volatility measurement.
    
    Args:
        data (pd.DataFrame): Stock data with OHLC columns
        window (int): Window size for ATR calculation
        
    Returns:
        pd.Series: ATR values
    """
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    # Calculate true range
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=window).mean()
    
    return atr


def detect_market_regime(data, long_window=50, short_window=10, atr_window=14, threshold=0.05):
    """
    Detect market regime based on trend and volatility indicators.
    
    Args:
        data (pd.DataFrame): Stock data with OHLC columns
        long_window (int): Window size for long-term trend detection
        short_window (int): Window size for short-term trend detection
        atr_window (int): Window size for ATR calculation
        threshold (float): Threshold for determining significant trend
        
    Returns:
        dict: Market regime information
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Calculate moving averages for trend detection
    df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
    
    # Calculate trend strength indicators
    df['Trend_Strength'] = (df['SMA_Short'] / df['SMA_Long'] - 1) * 100
    
    # Calculate volatility using ATR
    df['ATR'] = calculate_atr(df, window=atr_window)
    df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
    
    # Determine market regime for the current period
    if len(df) < max(long_window, atr_window) + 10:
        # Not enough data for reliable regime detection
        current_regime = "Unknown"
        regime_change = False
        confidence = 0.0
    else:
        # Get current values
        current_trend = df['Trend_Strength'].iloc[-1]
        recent_trend = df['Trend_Strength'].iloc[-10:].mean()
        current_atr = df['ATR_Pct'].iloc[-1]
        avg_atr = df['ATR_Pct'].iloc[-long_window:].mean()
        
        # Detect regime
        if abs(current_trend) < threshold:
            # Low trend strength indicates range-bound market
            current_regime = "Range-Bound"
            confidence = 1 - (abs(current_trend) / threshold)
        elif current_trend > threshold:
            # Positive trend strength indicates uptrend
            current_regime = "Trending Up"
            confidence = min(1.0, current_trend / (threshold * 3))
        else:
            # Negative trend strength indicates downtrend
            current_regime = "Trending Down"
            confidence = min(1.0, abs(current_trend) / (threshold * 3))
        
        # Determine if there was a recent regime change
        if len(df) >= long_window + 20:
            prev_trend = df['Trend_Strength'].iloc[-21:-11].mean()
            regime_change = (
                (recent_trend > threshold and prev_trend <= threshold) or
                (recent_trend < -threshold and prev_trend >= -threshold) or
                (abs(recent_trend) < threshold and abs(prev_trend) >= threshold)
            )
        else:
            regime_change = False
        
        # Adjust confidence based on volatility
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1
        if volatility_ratio > 1.5:
            # High volatility reduces confidence
            confidence = max(0.2, confidence * 0.8)
        
    # Calculate regime duration
    if current_regime == "Range-Bound":
        duration = sum(abs(df['Trend_Strength'].iloc[-long_window:]) < threshold)
    elif current_regime == "Trending Up":
        duration = sum(df['Trend_Strength'].iloc[-long_window:] > threshold)
    elif current_regime == "Trending Down":
        duration = sum(df['Trend_Strength'].iloc[-long_window:] < -threshold)
    else:
        duration = 0
    
    # Return regime information
    regime_info = {
        'regime': current_regime,
        'confidence': confidence,
        'regime_change': regime_change,
        'duration': duration,
        'data': df
    }
    
    return regime_info


def get_preferred_models_for_regime(regime):
    """
    Return preferred prediction models for different market regimes.
    
    Args:
        regime (str): Detected market regime
        
    Returns:
        list: List of preferred model names
    """
    if regime == "Trending Up":
        # In uptrends, trend-following models work best
        return ['Linear Regression', 'Time Series', 'ARIMA']
    elif regime == "Trending Down":
        # In downtrends, models that can capture acceleration work well
        return ['Linear Regression', 'Quadratic Regression', 'ARIMA']
    elif regime == "Range-Bound":
        # In range-bound markets, mean reversion models work well
        return ['Fourier Transform', 'Quadratic Regression']
    else:
        # Default to using all models
        return ['Linear Regression', 'Quadratic Regression', 'Fourier Transform', 'Time Series', 'ARIMA']


def plot_market_regime(data, regime_info):
    """
    Create a plot visualizing the market regime.
    
    Args:
        data (pd.DataFrame): Original stock data
        regime_info (dict): Market regime information
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with regime visualization
    """
    # Get data with regime indicators
    df = regime_info['data']
    regime = regime_info['regime']
    
    # Create subplots
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Price Chart", "Trend Strength", "Volatility (ATR%)"))
    
    # Add candlestick chart to first subplot
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add moving averages to first subplot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_Short'],
            mode='lines',
            name="Short-term SMA",
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_Long'],
            mode='lines',
            name="Long-term SMA",
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )
    
    # Add trend strength to second subplot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Trend_Strength'],
            mode='lines',
            name="Trend Strength",
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    # Add horizontal lines for trend strength thresholds
    threshold = 0.05  # Same as used in detect_market_regime
    fig.add_shape(
        type="line", x0=df.index[0], x1=df.index[-1], y0=threshold, y1=threshold,
        line=dict(color="rgba(0,255,0,0.5)", width=1, dash="dash"),
        row=2, col=1
    )
    
    fig.add_shape(
        type="line", x0=df.index[0], x1=df.index[-1], y0=-threshold, y1=-threshold,
        line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"),
        row=2, col=1
    )
    
    # Add ATR to third subplot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ATR_Pct'],
            mode='lines',
            name="ATR%",
            line=dict(color='purple')
        ),
        row=3, col=1
    )
    
    # Add regime annotation to the first subplot
    regime_colors = {
        "Trending Up": "rgba(0,255,0,0.2)",
        "Trending Down": "rgba(255,0,0,0.2)",
        "Range-Bound": "rgba(255,255,0,0.2)",
        "Unknown": "rgba(128,128,128,0.2)"
    }
    
    # Add colored background to indicate regime
    if len(df) > 0:
        fig.add_shape(
            type="rect",
            x0=df.index[-min(len(df), 20)],
            x1=df.index[-1],
            y0=data['Low'].min(),
            y1=data['High'].max(),
            fillcolor=regime_colors.get(regime, "rgba(128,128,128,0.2)"),
            opacity=0.5,
            layer="below",
            line_width=0,
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"Market Regime Analysis: {regime}",
        xaxis_title="Date",
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Trend Strength (%)", row=2, col=1)
    fig.update_yaxes(title_text="ATR (%)", row=3, col=1)
    
    # Remove rangeslider from candlestick chart
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    
    return fig


def get_regime_description(regime):
    """
    Get a detailed description of the market regime.
    
    Args:
        regime (str): Detected market regime
        
    Returns:
        str: Description of the market regime
    """
    descriptions = {
        "Trending Up": """
        **Trending Up Market**
        
        In an uptrend, prices make higher highs and higher lows. This indicates strong buying pressure 
        and positive market sentiment. Trend-following strategies like moving average systems tend to 
        work well in this regime. Linear and time series models often perform better in trending markets.
        
        **Trading Characteristics:**
        - Support and resistance levels tend to be respected
        - Pullbacks are typically shallow
        - Momentum indicators show strong readings
        
        **Prediction Models Used:**
        - Linear Regression (captures trend direction)
        - Time Series (captures sequential patterns)
        - ARIMA (captures autoregressive trend components)
        """,
        
        "Trending Down": """
        **Trending Down Market**
        
        In a downtrend, prices make lower highs and lower lows. This indicates strong selling pressure 
        and negative market sentiment. Models that can capture acceleration and volatility tend to work 
        well in downtrends, as prices often fall faster than they rise.
        
        **Trading Characteristics:**
        - Resistance levels tend to be stronger than support
        - Rallies are typically short-lived
        - Volatility is often higher than in uptrends
        
        **Prediction Models Used:**
        - Linear Regression (captures trend direction)
        - Quadratic Regression (captures accelerating downward moves)
        - ARIMA (captures autoregressive trend components)
        """,
        
        "Range-Bound": """
        **Range-Bound Market**
        
        In a range-bound or sideways market, prices oscillate between clear support and resistance levels. 
        This indicates balanced buying and selling pressure with no clear directional bias. Mean-reversion 
        strategies tend to work best in this regime.
        
        **Trading Characteristics:**
        - Strong horizontal support and resistance levels
        - Price oscillates predictably within a range
        - Breakouts from the range often signal regime change
        
        **Prediction Models Used:**
        - Fourier Transform (captures cyclical patterns)
        - Quadratic Regression (captures curved price movements)
        """,
        
        "Unknown": """
        **Unknown Market Regime**
        
        There is not enough data to determine the current market regime with confidence. All prediction
        models will be used with equal weighting until a clear regime emerges.
        
        **Prediction Approach:**
        - Use ensemble of all models with equal weighting
        - Consider volatility and recent price action for trading decisions
        """
    }
    
    return descriptions.get(regime, "No description available for this regime.")


def get_recommended_settings(regime):
    """
    Get recommended prediction settings based on market regime.
    
    Args:
        regime (str): Detected market regime
        
    Returns:
        dict: Recommended prediction settings
    """
    if regime == "Trending Up":
        return {
            'prediction_days': 30,  # Trends can persist
            'model_weights': {
                'Linear Regression': 0.35,
                'Time Series': 0.35,
                'ARIMA': 0.20,
                'Quadratic Regression': 0.05,
                'Fourier Transform': 0.05
            }
        }
    elif regime == "Trending Down":
        return {
            'prediction_days': 20,  # Downtrends can reverse more quickly
            'model_weights': {
                'Linear Regression': 0.30,
                'Quadratic Regression': 0.30,
                'ARIMA': 0.25,
                'Time Series': 0.10,
                'Fourier Transform': 0.05
            }
        }
    elif regime == "Range-Bound":
        return {
            'prediction_days': 15,  # Range-bound markets more suitable for shorter horizons
            'model_weights': {
                'Fourier Transform': 0.40,
                'Quadratic Regression': 0.30,
                'ARIMA': 0.15,
                'Time Series': 0.10,
                'Linear Regression': 0.05
            }
        }
    else:
        return {
            'prediction_days': 20,  # Balanced approach
            'model_weights': {
                'Linear Regression': 0.20,
                'Quadratic Regression': 0.20,
                'Fourier Transform': 0.20,
                'Time Series': 0.20,
                'ARIMA': 0.20
            }
        }