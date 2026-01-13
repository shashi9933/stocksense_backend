import os
import re
import streamlit as st
import pandas as pd
from datetime import datetime

class PriceAlert:
    def __init__(self, stock_symbol, target_price, direction, phone_number, creation_time=None):
        """
        Initialize a price alert object.
        
        Args:
            stock_symbol (str): Stock symbol
            target_price (float): Target price for alert
            direction (str): 'above' or 'below'
            phone_number (str): Phone number to send alert to
            creation_time (datetime, optional): Time alert was created
        """
        self.stock_symbol = stock_symbol
        self.target_price = target_price
        self.direction = direction
        self.phone_number = phone_number
        self.creation_time = creation_time or datetime.now()
        self.triggered = False
    
    def to_dict(self):
        """Convert alert to dictionary for storage."""
        return {
            'stock_symbol': self.stock_symbol,
            'target_price': self.target_price,
            'direction': self.direction,
            'phone_number': self.phone_number,
            'creation_time': self.creation_time,
            'triggered': self.triggered
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create alert from dictionary."""
        alert = cls(
            data['stock_symbol'],
            data['target_price'],
            data['direction'],
            data['phone_number'],
            data['creation_time']
        )
        alert.triggered = data['triggered']
        return alert

def validate_phone_number(phone_number):
    """
    Validate international phone number format.
    
    Args:
        phone_number (str): Phone number to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Basic validation for international format (start with + and followed by digits)
    pattern = r'^\+\d{1,3}\d{6,14}$'
    return bool(re.match(pattern, phone_number))

def check_price_alerts(stock_data, alerts):
    """
    Check if any price alerts should be triggered.
    
    Args:
        stock_data (pd.DataFrame): Stock data with latest prices
        alerts (list): List of PriceAlert objects
        
    Returns:
        tuple: (triggered_alerts, updated_alerts)
    """
    if stock_data is None or stock_data.empty:
        return [], alerts
    
    current_price = stock_data['Close'].iloc[-1]
    triggered_alerts = []
    updated_alerts = []
    
    for alert in alerts:
        if alert.triggered:
            updated_alerts.append(alert)
            continue
        
        if (alert.direction == 'above' and current_price >= alert.target_price) or \
           (alert.direction == 'below' and current_price <= alert.target_price):
            alert.triggered = True
            triggered_alerts.append(alert)
        
        updated_alerts.append(alert)
    
    return triggered_alerts, updated_alerts

def send_alert_notification(alert):
    """
    Display an in-app notification for a triggered alert.
    
    Args:
        alert (PriceAlert): Triggered alert to send
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create notification message
        message_text = (
            f"Price Alert for {alert.stock_symbol}: "
            f"Price has moved {alert.direction} your target of ${alert.target_price:.2f}."
        )
        
        # Display in-app notification
        st.success(message_text)
        
        return True
    except Exception as e:
        st.error(f"Failed to display alert: {str(e)}")
        return False

def get_active_alerts():
    """Get active alerts from session state."""
    if 'active_alerts' not in st.session_state:
        st.session_state.active_alerts = []
    
    return st.session_state.active_alerts

def add_price_alert(stock_symbol, target_price, direction, phone_number):
    """
    Add a new price alert.
    
    Args:
        stock_symbol (str): Stock symbol
        target_price (float): Target price for alert
        direction (str): 'above' or 'below'
        phone_number (str): Phone number to send alert to
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not validate_phone_number(phone_number):
        return False
    
    alert = PriceAlert(stock_symbol, target_price, direction, phone_number)
    
    if 'active_alerts' not in st.session_state:
        st.session_state.active_alerts = []
    
    st.session_state.active_alerts.append(alert)
    return True

def remove_price_alert(index):
    """
    Remove a price alert by index.
    
    Args:
        index (int): Index of alert to remove
    """
    if 'active_alerts' in st.session_state and 0 <= index < len(st.session_state.active_alerts):
        st.session_state.active_alerts.pop(index)

def display_alerts():
    """
    Display active alerts in a tabular format.
    
    Returns:
        list: List of indices for alerts to remove
    """
    if 'active_alerts' not in st.session_state or not st.session_state.active_alerts:
        st.info("No active price alerts.")
        return []
    
    alerts = st.session_state.active_alerts
    
    # Create a table of alerts
    alert_data = []
    for i, alert in enumerate(alerts):
        alert_data.append({
            'Symbol': alert.stock_symbol,
            'Target Price': f"${alert.target_price:.2f}",
            'Direction': alert.direction.capitalize(),
            'Phone Number': alert.phone_number,
            'Created': alert.creation_time.strftime('%Y-%m-%d %H:%M'),
            'Status': 'Triggered' if alert.triggered else 'Active'
        })
    
    if alert_data:
        df = pd.DataFrame(alert_data)
        st.table(df)
    
    # Add checkboxes to select alerts to remove
    st.subheader("Remove Alerts")
    indices_to_remove = []
    
    for i, alert in enumerate(alerts):
        if st.checkbox(f"Remove alert for {alert.stock_symbol} @ ${alert.target_price:.2f}", key=f"remove_{i}"):
            indices_to_remove.append(i)
    
    return indices_to_remove
