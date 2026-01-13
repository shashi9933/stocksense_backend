import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_sma(data, window=20):
    """
    Calculate Simple Moving Average.
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        window (int): Window size for moving average
        
    Returns:
        pd.Series: Series containing SMA values
    """
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, window=20):
    """
    Calculate Exponential Moving Average.
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        window (int): Window size for moving average
        
    Returns:
        pd.Series: Series containing EMA values
    """
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index.
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        window (int): Window size for RSI
        
    Returns:
        pd.Series: Series containing RSI values
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # For the first window observations, RSI is not defined
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        window (int): Window size for moving average
        num_std (int): Number of standard deviations
        
    Returns:
        tuple: (Upper band, Middle band, Lower band)
    """
    middle_band = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    # Calculate fast and slow EMAs
    fast_ema = data['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line (EMA of MACD line)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram (MACD line - Signal line)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_support_resistance(data, window=10):
    """
    Calculate support and resistance levels.
    
    Args:
        data (pd.DataFrame): Stock data with 'High' and 'Low' columns
        window (int): Window size for local min/max detection
        
    Returns:
        tuple: (support_levels, resistance_levels)
    """
    # Identify local maxima for resistance levels
    resistance_levels = []
    for i in range(window, len(data) - window):
        if all(data['High'].iloc[i] > data['High'].iloc[i-j] for j in range(1, window+1)) and \
           all(data['High'].iloc[i] > data['High'].iloc[i+j] for j in range(1, window+1)):
            resistance_levels.append((data.index[i], data['High'].iloc[i]))
    
    # Identify local minima for support levels
    support_levels = []
    for i in range(window, len(data) - window):
        if all(data['Low'].iloc[i] < data['Low'].iloc[i-j] for j in range(1, window+1)) and \
           all(data['Low'].iloc[i] < data['Low'].iloc[i+j] for j in range(1, window+1)):
            support_levels.append((data.index[i], data['Low'].iloc[i]))
    
    return support_levels, resistance_levels

def detect_candlestick_patterns(data):
    """
    Detect common candlestick patterns.
    
    Args:
        data (pd.DataFrame): Stock data with OHLC columns
        
    Returns:
        dict: Dictionary of detected patterns
    """
    patterns = {}
    
    # Detect Doji patterns
    doji = []
    for i in range(len(data)):
        if abs(data['Open'].iloc[i] - data['Close'].iloc[i]) / (data['High'].iloc[i] - data['Low'].iloc[i] + 0.001) < 0.1:
            if data['High'].iloc[i] - max(data['Open'].iloc[i], data['Close'].iloc[i]) > 3 * abs(data['Open'].iloc[i] - data['Close'].iloc[i]) and \
               min(data['Open'].iloc[i], data['Close'].iloc[i]) - data['Low'].iloc[i] > 3 * abs(data['Open'].iloc[i] - data['Close'].iloc[i]):
                doji.append((data.index[i], 'Long-Legged Doji'))
            else:
                doji.append((data.index[i], 'Doji'))
    
    # Detect Hammer patterns
    hammer = []
    for i in range(len(data)):
        body = abs(data['Open'].iloc[i] - data['Close'].iloc[i])
        if body > 0:
            lower_shadow = min(data['Open'].iloc[i], data['Close'].iloc[i]) - data['Low'].iloc[i]
            upper_shadow = data['High'].iloc[i] - max(data['Open'].iloc[i], data['Close'].iloc[i])
            if lower_shadow > 2 * body and upper_shadow < 0.1 * body:
                hammer.append((data.index[i], 'Hammer'))
    
    # Detect Engulfing patterns
    engulfing = []
    for i in range(1, len(data)):
        # Bullish engulfing
        if data['Close'].iloc[i-1] < data['Open'].iloc[i-1] and \
           data['Close'].iloc[i] > data['Open'].iloc[i] and \
           data['Close'].iloc[i] > data['Open'].iloc[i-1] and \
           data['Open'].iloc[i] < data['Close'].iloc[i-1]:
            engulfing.append((data.index[i], 'Bullish Engulfing'))
        
        # Bearish engulfing
        elif data['Close'].iloc[i-1] > data['Open'].iloc[i-1] and \
             data['Close'].iloc[i] < data['Open'].iloc[i] and \
             data['Close'].iloc[i] < data['Open'].iloc[i-1] and \
             data['Open'].iloc[i] > data['Close'].iloc[i-1]:
            engulfing.append((data.index[i], 'Bearish Engulfing'))
    
    patterns['doji'] = doji
    patterns['hammer'] = hammer
    patterns['engulfing'] = engulfing
    
    return patterns

def plot_with_indicators(data, indicators):
    """
    Create a plot with specified technical indicators.
    
    Args:
        data (pd.DataFrame): Stock data with OHLC columns
        indicators (list): List of indicators to include
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with indicators
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1, 
        row_heights=[0.7, 0.3],
        subplot_titles=("Price", "Indicators")
    )
    
    # Add candlestick chart
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
    
    # Add selected indicators
    if 'SMA' in indicators:
        periods = [20, 50, 200]
        colors = ['rgba(13, 71, 161, 0.7)', 'rgba(46, 125, 50, 0.7)', 'rgba(183, 28, 28, 0.7)']
        
        for period, color in zip(periods, colors):
            sma = calculate_sma(data, window=period)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=sma,
                    line=dict(color=color, width=1),
                    name=f"SMA {period}"
                ),
                row=1, col=1
            )
    
    if 'Bollinger Bands' in indicators:
        upper, middle, lower = calculate_bollinger_bands(data)
        
        # Add middle band
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=middle,
                line=dict(color='rgba(46, 125, 50, 0.7)', width=1),
                name="BB Middle"
            ),
            row=1, col=1
        )
        
        # Add upper and lower bands
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=upper,
                line=dict(color='rgba(0, 0, 0, 0)'),
                name="BB Upper"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=lower,
                line=dict(color='rgba(0, 0, 0, 0)'),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.2)',
                name="BB Lower"
            ),
            row=1, col=1
        )
    
    if 'RSI' in indicators:
        rsi = calculate_rsi(data)
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=rsi,
                line=dict(color='purple', width=1),
                name="RSI"
            ),
            row=2, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_shape(
            type='line',
            x0=data.index[0],
            y0=70,
            x1=data.index[-1],
            y1=70,
            line=dict(color='red', width=1, dash='dash'),
            row=2, col=1
        )
        
        fig.add_shape(
            type='line',
            x0=data.index[0],
            y0=30,
            x1=data.index[-1],
            y1=30,
            line=dict(color='green', width=1, dash='dash'),
            row=2, col=1
        )
        
        # Update y-axis range for RSI
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    
    if 'MACD' in indicators:
        macd_line, signal_line, histogram = calculate_macd(data)
        
        # Add MACD line and signal line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=macd_line,
                line=dict(color='blue', width=1),
                name="MACD"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=signal_line,
                line=dict(color='red', width=1),
                name="Signal"
            ),
            row=2, col=1
        )
        
        # Add histogram as bar chart
        colors = ['green' if val >= 0 else 'red' for val in histogram]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=histogram,
                marker_color=colors,
                name="Histogram"
            ),
            row=2, col=1
        )
    
    # Add support and resistance lines if selected
    if 'Support/Resistance' in indicators:
        support_levels, resistance_levels = calculate_support_resistance(data)
        
        # Add resistance levels
        for date, level in resistance_levels:
            fig.add_shape(
                type="line",
                x0=date,
                y0=level,
                x1=data.index[-1],
                y1=level,
                line=dict(color="red", width=1, dash="dash"),
                row=1, col=1
            )
        
        # Add support levels
        for date, level in support_levels:
            fig.add_shape(
                type="line",
                x0=date,
                y0=level,
                x1=data.index[-1],
                y1=level,
                line=dict(color="green", width=1, dash="dash"),
                row=1, col=1
            )
    
    # Update layout
    fig.update_layout(
        title_text="Technical Analysis Chart",
        height=800,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Indicator Value", row=2, col=1)
    
    return fig
