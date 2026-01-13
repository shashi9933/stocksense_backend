import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_candlestick_chart(data, title=None, height=700, show_volume=True):
    """
    Create a professional candlestick chart with optional volume bars.
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV columns
        title (str, optional): Chart title
        height (int): Chart height
        show_volume (bool): Whether to show volume bars
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if show_volume:
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.8, 0.2]
        )
    else:
        fig = go.Figure()
    
    # Add candlestick chart
    candlestick = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add volume bars if requested
    if show_volume and 'Volume' in data.columns:
        colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in data.iterrows()]
        
        volume_bars = go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=colors,
            name="Volume"
        )
        
        fig.add_trace(volume_bars, row=2, col=1)
    
    # Update layout
    layout_args = {
        'xaxis_rangeslider_visible': False,
        'template': 'plotly_white',
        'height': height,
        'hovermode': 'x unified'
    }
    
    if title:
        layout_args['title'] = title
    
    fig.update_layout(**layout_args)
    
    if show_volume:
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Price")
    
    return fig

def create_range_selector():
    """
    Create a range selector for charts.
    
    Returns:
        dict: Range selector configuration
    """
    return dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )

def add_range_selector(fig, row=1, col=1):
    """
    Add a range selector to an existing figure.
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure
        row (int): Row to add the selector to
        col (int): Column to add the selector to
        
    Returns:
        plotly.graph_objects.Figure: Updated figure
    """
    rangeselector = create_range_selector()
    
    fig.update_xaxes(
        rangeselector=rangeselector,
        row=row,
        col=col
    )
    
    return fig

def add_annotations(fig, data, annotations, row=1, col=1):
    """
    Add annotation markers to a chart.
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure
        data (pd.DataFrame): Stock data
        annotations (dict): Dictionary of annotations with dates and labels
        row (int): Row to add annotations to
        col (int): Column to add annotations to
        
    Returns:
        plotly.graph_objects.Figure: Updated figure
    """
    for date, label in annotations:
        if date in data.index:
            y_position = data.loc[date, 'High'] * 1.02  # Position slightly above the high
            
            fig.add_annotation(
                x=date,
                y=y_position,
                text=label,
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                row=row,
                col=col
            )
    
    return fig

def calculate_pivot_points(data):
    """
    Calculate classic pivot points based on the last trading day.
    
    Args:
        data (pd.DataFrame): Stock data with OHLC columns
        
    Returns:
        dict: Dictionary of pivot points
    """
    # Get the last trading day data
    last_day = data.iloc[-1]
    high = last_day['High']
    low = last_day['Low']
    close = last_day['Close']
    
    # Calculate pivot point
    pivot = (high + low + close) / 3
    
    # Calculate support and resistance levels
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    return {
        'P': pivot,
        'S1': s1,
        'S2': s2,
        'S3': s3,
        'R1': r1,
        'R2': r2,
        'R3': r3
    }

def add_pivot_points(fig, data, row=1, col=1):
    """
    Add pivot points to a chart.
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure
        data (pd.DataFrame): Stock data
        row (int): Row to add pivot points to
        col (int): Column to add pivot points to
        
    Returns:
        plotly.graph_objects.Figure: Updated figure
    """
    # Calculate pivot points
    pivot_points = calculate_pivot_points(data)
    
    # Define line colors for different levels
    colors = {
        'P': 'black',
        'S1': 'green',
        'S2': 'green',
        'S3': 'green',
        'R1': 'red',
        'R2': 'red',
        'R3': 'red'
    }
    
    # Define line styles
    dash_styles = {
        'P': 'solid',
        'S1': 'dash',
        'S2': 'dash',
        'S3': 'dash',
        'R1': 'dash',
        'R2': 'dash',
        'R3': 'dash'
    }
    
    # Add horizontal lines for each pivot point
    for level, value in pivot_points.items():
        fig.add_shape(
            type="line",
            x0=data.index[0],
            y0=value,
            x1=data.index[-1],
            y1=value,
            line=dict(
                color=colors[level],
                width=1,
                dash=dash_styles[level]
            ),
            row=row,
            col=col
        )
        
        # Add text labels
        fig.add_annotation(
            x=data.index[-1],
            y=value,
            text=f"{level}: {value:.2f}",
            showarrow=False,
            xanchor="left",
            xshift=10,
            row=row,
            col=col
        )
    
    return fig

def format_number(number):
    """
    Format numbers for display (e.g., adding commas, K, M, B suffixes).
    
    Args:
        number (float): Number to format
        
    Returns:
        str: Formatted number
    """
    abs_num = abs(number)
    sign = -1 if number < 0 else 1
    
    if abs_num >= 1_000_000_000:
        return f"{sign * abs_num / 1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        return f"{sign * abs_num / 1_000_000:.2f}M"
    elif abs_num >= 1_000:
        return f"{sign * abs_num / 1_000:.2f}K"
    else:
        return f"{number:,.2f}"
