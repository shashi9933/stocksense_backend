import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as poly
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.fft import fft, ifft
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def create_features(data, window_sizes=[5, 10, 20, 30]):
    """
    Create technical features for prediction models.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
        window_sizes (list): List of window sizes for rolling features
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    if len(data) < max(window_sizes) + 10:
        # Not enough data points for reliable feature calculation
        raise ValueError(f"Insufficient data: need at least {max(window_sizes) + 10} data points")
        
    df = data.copy()
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
    
    # Add moving averages
    for window in window_sizes:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Add moving average crossover signals
        if window < 20:
            larger_window = window * 2
            df[f'SMA_Cross_{window}_{larger_window}'] = (
                df[f'SMA_{window}'] > df[f'SMA_{larger_window}']
            ).astype(int)
    
    # Add rolling statistics
    for window in window_sizes:
        df[f'Std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'Max_{window}'] = df['Close'].rolling(window=window).max()
        df[f'Median_{window}'] = df['Close'].rolling(window=window).median()
        df[f'Skew_{window}'] = df['Close'].rolling(window=window).skew()
        
        # Price relative to its moving range
        df[f'Price_Range_Pct_{window}'] = (df['Close'] - df[f'Min_{window}']) / (df[f'Max_{window}'] - df[f'Min_{window}'])
    
    # Add price momentum
    for window in window_sizes:
        df[f'Momentum_{window}'] = df['Close'] - df['Close'].shift(window)
        df[f'Rate_Of_Change_{window}'] = df['Close'].pct_change(window)
    
    # Add volume features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_to_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    for window in window_sizes:
        df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window=window).mean()
        df[f'Volume_Std_{window}'] = df['Volume'].rolling(window=window).std()
        
        # Price-volume relationship features
        df[f'Price_Volume_Corr_{window}'] = (
            df['Close'].rolling(window=window).corr(df['Volume'])
        )
    
    # Add volatility
    df['Daily_Range'] = df['High'] - df['Low']
    df['Daily_Range_Pct'] = df['Daily_Range'] / df['Close']
    
    for window in window_sizes:
        df[f'Range_SMA_{window}'] = df['Daily_Range'].rolling(window=window).mean()
        df[f'Range_Std_{window}'] = df['Daily_Range'].rolling(window=window).std()
        
        # Normalized price variance
        df[f'Close_Normalized_{window}'] = df['Close'] / df[f'SMA_{window}'] - 1
        
    # Add trend strength indicators    
    df['Up_Down_Ratio_10'] = np.where(df['Returns'] > 0, 1, 0).rolling(10).sum() / 10
    df['Up_Down_Ratio_20'] = np.where(df['Returns'] > 0, 1, 0).rolling(20).sum() / 20
    
    # OHLC relationships
    df['Close_to_Open'] = df['Close'] / df['Open'] - 1
    df['High_to_Low'] = df['High'] / df['Low'] - 1
    df['Mid_Point'] = (df['High'] + df['Low']) / 2
    df['Close_to_Mid'] = df['Close'] / df['Mid_Point'] - 1
    
    # Time-based features for potential seasonality
    if not df.index.empty and isinstance(df.index[0], pd.Timestamp):
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Day_of_Month'] = df.index.day
        df['Quarter'] = df.index.quarter
        
        # Use sine and cosine transformations for cyclical features
        df['Day_of_Week_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_of_Week_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Drop rows with NaN values (from rolling calculations)
    df = df.dropna()
    
    return df

def linear_regression_prediction(data, prediction_days=30):
    """
    Enhanced linear regression prediction model with cross-validation and regularization.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
        prediction_days (int): Number of days to predict forward
        
    Returns:
        tuple: (predictions, confidence)
    """
    try:
        # Create features
        df = create_features(data)
        
        # Create target variable (next day's close price)
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        # Select features and target
        features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
        X = df[features]
        y = df['Target']
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Prepare time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train and evaluate multiple models to find the best one
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1)
        }
        
        best_model = None
        best_score = -float('inf')
        
        for name, model in models.items():
            # Perform time series cross-validation
            cv_scores = []
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                cv_scores.append(model.score(X_test, y_test))
            
            avg_score = np.mean(cv_scores)
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # If no model performed well, fall back to standard LinearRegression
        if best_model is None or best_score < 0:
            best_model = LinearRegression()
            best_model.fit(X_scaled, y)
            best_score = best_model.score(X_scaled, y)
        else:
            # Retrain the best model on all data
            best_model.fit(X_scaled, y)
        
        # Feature importance (if available)
        if hasattr(best_model, 'coef_'):
            # Get feature importance
            importance = np.abs(best_model.coef_)
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Keep only the most important features for prediction
            top_features = feature_importance['Feature'].head(min(20, len(features))).tolist()
            top_indices = [features.index(f) for f in top_features]
        else:
            top_indices = range(len(features))
            
        # Prepare prediction data
        last_data_point = X.iloc[-1].values.reshape(1, -1)
        last_data_point_scaled = scaler.transform(last_data_point)
        
        # Make predictions
        predictions = []
        confidence = []
        
        # Calculate prediction error based on cross-validation
        cv_error = 1.0 - best_score
        
        for i in range(prediction_days):
            # Update features with last prediction (this is simplified as we don't have a way to update all features)
            prediction_input = last_data_point_scaled.copy()
            next_pred = best_model.predict(prediction_input)[0]
            
            # Ensure prediction is non-negative
            next_pred = max(0, next_pred)
            
            # Store prediction
            predictions.append(next_pred)
            
            # Calculate confidence based on cross-validation score and distance into the future
            # Confidence decreases the further we predict
            time_decay = np.exp(-0.05 * i)  # Exponential decay with time
            day_confidence = max(0, best_score * time_decay)
            confidence.append(day_confidence)
        
        return predictions, confidence
    
    except Exception as e:
        # Fallback to a simpler model if feature creation fails
        st.warning(f"Advanced linear model failed: {e}. Using simple trend model.")
        
        # Simple trend-based prediction as fallback
        prices = data['Close'].values
        
        # Calculate average price change
        avg_change = np.mean(np.diff(prices[-min(30, len(prices)-1):]))
        
        # Generate predictions based on trend
        last_price = prices[-1]
        predictions = [max(0, last_price + avg_change * (i+1)) for i in range(prediction_days)]
        
        # Lower confidence for the fallback model
        confidence = [0.3 * np.exp(-0.05 * i) for i in range(prediction_days)]
        
        return predictions, confidence

def quadratic_regression_prediction(data, prediction_days=30):
    """
    Quadratic regression prediction model.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
        prediction_days (int): Number of days to predict forward
        
    Returns:
        tuple: (predictions, confidence)
    """
    # Use price data for polynomial regression
    y = data['Close'].values
    x = np.arange(len(y))
    
    # Fit quadratic polynomial
    coeffs = poly.polyfit(x, y, 2)
    
    # Calculate R-squared for confidence
    model = poly.Polynomial(coeffs)
    y_pred = model(x)
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Predict future values
    future_x = np.arange(len(y), len(y) + prediction_days)
    predictions = model(future_x)
    
    # Use R-squared as confidence
    confidence = [r_squared] * prediction_days
    
    return predictions, confidence

def fourier_transform_prediction(data, prediction_days=30, harmonics=10):
    """
    Fourier transform prediction model.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
        prediction_days (int): Number of days to predict forward
        harmonics (int): Number of harmonics to include
        
    Returns:
        tuple: (predictions, confidence)
    """
    # Get close prices
    prices = data['Close'].values
    
    # Compute FFT
    fft_result = fft(prices)
    
    # Keep only the most significant harmonics
    fft_result_filtered = np.copy(fft_result)
    fft_result_filtered[harmonics:-harmonics] = 0
    
    # Inverse FFT to get the filtered time series
    filtered_prices = ifft(fft_result_filtered).real
    
    # Calculate R-squared for confidence
    ss_tot = np.sum((prices - np.mean(prices))**2)
    ss_res = np.sum((prices - filtered_prices)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Extend the time series
    n = len(prices)
    extended_time = np.arange(0, n + prediction_days)
    t = np.arange(0, n)
    
    # Fit a polynomial to the filtered data
    degree = min(5, len(prices) // 10)  # Avoid overfitting
    poly_coeffs = np.polyfit(t, filtered_prices, degree)
    poly_model = np.poly1d(poly_coeffs)
    
    # Generate predictions
    predictions = poly_model(np.arange(n, n + prediction_days))
    
    # Adjust confidence based on length of prediction
    confidence = [r_squared * np.exp(-0.05 * i) for i in range(prediction_days)]
    
    return predictions, confidence

def arima_model_prediction(data, prediction_days=30):
    """
    ARIMA time series prediction model.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
        prediction_days (int): Number of days to predict forward
        
    Returns:
        tuple: (predictions, confidence)
    """
    try:
        # Get close prices
        prices = data['Close'].values
        
        # Check for stationarity
        result = adfuller(prices)
        p_value = result[1]
        is_stationary = p_value < 0.05
        
        # Prepare data
        if not is_stationary:
            # Take first difference if not stationary
            prices_diff = np.diff(prices)
            d_order = 1
        else:
            prices_diff = prices
            d_order = 0
        
        # Try different ARIMA orders
        best_aic = float('inf')
        best_order = None
        
        # Limit search to prevent excessive computation
        p_values = range(0, 3)
        q_values = range(0, 3)
        
        for p in p_values:
            for q in q_values:
                try:
                    model = ARIMA(prices, order=(p, d_order, q))
                    model_fit = model.fit()
                    
                    # Calculate AIC
                    aic = model_fit.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d_order, q)
                except:
                    continue
        
        # If no model was found, use default order
        if best_order is None:
            best_order = (1, d_order, 1)
        
        # Fit the best model
        final_model = ARIMA(prices, order=best_order)
        final_model_fit = final_model.fit()
        
        # Make predictions
        forecast = final_model_fit.forecast(steps=prediction_days)
        predictions = forecast
        
        # Calculate confidence based on model fit
        # Since we don't have true future values, we'll approximate using in-sample fit
        in_sample_preds = final_model_fit.predict(start=1, end=len(prices))
        in_sample_rmse = np.sqrt(mean_squared_error(prices[1:], in_sample_preds[1:]))
        
        # Normalize RMSE to get a confidence score between 0 and 1
        max_price = np.max(prices)
        normalized_rmse = min(1, max(0, 1 - (in_sample_rmse / max_price)))
        
        # Confidence decreases with forecast horizon
        confidence = [normalized_rmse * np.exp(-0.05 * i) for i in range(prediction_days)]
        
        return predictions, confidence
    
    except Exception as e:
        # Fallback to simple exponential smoothing if ARIMA fails
        st.warning(f"ARIMA model failed: {e}. Using exponential smoothing.")
        
        try:
            # Get the prices here inside the exception handler to ensure it's defined
            prices = data['Close'].values
            
            # Simple exponential smoothing as fallback
            alpha = 0.3  # Smoothing factor
            last_price = prices[-1]
            
            # Calculate average price change over last 10 days
            if len(prices) >= 10:
                avg_change = np.mean(np.diff(prices[-10:]))
            else:
                avg_change = 0
            
            # Generate predictions with exponential trend
            predictions = []
            for i in range(prediction_days):
                next_price = last_price + avg_change * (1 - alpha)**(i)
                predictions.append(max(0, next_price))  # Ensure non-negative
            
            # Low confidence for fallback model
            confidence = [0.3 * np.exp(-0.05 * i) for i in range(prediction_days)]
            
            return predictions, confidence
            
        except Exception as inner_e:
            # Ultimate fallback if even the simple model fails
            st.warning(f"Even fallback model failed: {inner_e}. Using constant price.")
            
            # Just return the last known price as a constant prediction
            last_price = data['Close'].iloc[-1]
            predictions = [last_price] * prediction_days
            confidence = [0.1] * prediction_days
            
            return predictions, confidence

def time_series_prediction(data, prediction_days=30):
    """
    Time series prediction model combining traditional and ARIMA approaches.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
        prediction_days (int): Number of days to predict forward
        
    Returns:
        tuple: (predictions, confidence)
    """
    try:
        # Get both ARIMA and regression-based predictions
        arima_preds, arima_conf = arima_model_prediction(data, prediction_days)
        
        # Create features
        df = create_features(data)
        
        # Create target variable (next day's close price)
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        # Select features and target
        features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
        X = df[features]
        y = df['Target']
        
        # Create lagged features to capture time series patterns
        for i in range(1, 6):
            X[f'Close_Lag_{i}'] = df['Close'].shift(i)
        
        # Drop rows with NaN values
        X = X.dropna()
        y = y.iloc[X.index]
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train model with ridge regularization to prevent overfitting
        model = Ridge(alpha=0.5)
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            cv_scores.append(model.score(X_test, y_test))
        
        # Retrain on all data
        model.fit(X_scaled, y)
        
        # Make regression predictions
        regression_preds = []
        regression_conf = []
        last_x = X.iloc[-1:].copy()
        
        for i in range(prediction_days):
            # Update lagged features for prediction
            for lag in range(5, 0, -1):
                if i >= lag:
                    last_x[f'Close_Lag_{lag}'] = regression_preds[i - lag]
                elif lag > 1:
                    last_x[f'Close_Lag_{lag}'] = last_x[f'Close_Lag_{lag-1}']
            
            # Scale the input
            x_scaled = scaler.transform(last_x)
            
            # Predict next value
            pred = model.predict(x_scaled)[0]
            regression_preds.append(pred)
            
            # Calculate confidence based on cross-validation score and time
            reg_conf = max(0, np.mean(cv_scores)) * np.exp(-0.05 * i)
            regression_conf.append(reg_conf)
        
        # Combine predictions from ARIMA and regression models
        # Weight them based on confidence
        predictions = []
        confidence = []
        
        for i in range(prediction_days):
            # Get confidence values for this day
            arima_confidence = arima_conf[i]
            reg_confidence = regression_conf[i]
            
            # Calculate weights
            total_conf = arima_confidence + reg_confidence
            if total_conf > 0:
                arima_weight = arima_confidence / total_conf
                reg_weight = reg_confidence / total_conf
            else:
                arima_weight = reg_weight = 0.5
            
            # Calculate weighted prediction
            weighted_pred = (arima_preds[i] * arima_weight) + (regression_preds[i] * reg_weight)
            
            # Store results
            predictions.append(weighted_pred)
            confidence.append(max(arima_confidence, reg_confidence))
        
        return predictions, confidence
    
    except Exception as e:
        # Fallback to ARIMA only if the combined approach fails
        st.warning(f"Combined time series model failed: {e}. Using ARIMA only.")
        return arima_model_prediction(data, prediction_days)

def ensemble_prediction(data, prediction_days=30, regime_weights=None):
    """
    Ensemble prediction using multiple models with dynamic weighting based on market regime.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
        prediction_days (int): Number of days to predict forward
        regime_weights (dict, optional): Model weights based on market regime
        
    Returns:
        tuple: (predictions, confidence, model_weights)
    """
    try:
        # Import market regime detection if needed
        from utils.market_regime import detect_market_regime, get_recommended_settings
        
        # Detect market regime for adaptive model selection
        regime_info = detect_market_regime(data)
        current_regime = regime_info['regime']
        
        # Get recommended settings based on market regime if not provided
        if regime_weights is None:
            recommended_settings = get_recommended_settings(current_regime)
            regime_weights = recommended_settings['model_weights']
        
        # Get predictions from individual models - wrap each in try-except to ensure robustness
        try:
            linear_preds, linear_conf = linear_regression_prediction(data, prediction_days)
        except Exception as e:
            st.warning(f"Linear regression model failed: {e}. Using fallback.")
            # Simple fallback
            linear_preds = [data['Close'].iloc[-1]] * prediction_days
            linear_conf = [0.2] * prediction_days
        
        try:
            quad_preds, quad_conf = quadratic_regression_prediction(data, prediction_days)
        except Exception as e:
            st.warning(f"Quadratic regression model failed: {e}. Using fallback.")
            # Simple fallback
            quad_preds = [data['Close'].iloc[-1]] * prediction_days
            quad_conf = [0.2] * prediction_days
        
        try:
            fourier_preds, fourier_conf = fourier_transform_prediction(data, prediction_days)
        except Exception as e:
            st.warning(f"Fourier transform model failed: {e}. Using fallback.")
            # Simple fallback
            fourier_preds = [data['Close'].iloc[-1]] * prediction_days
            fourier_conf = [0.2] * prediction_days
        
        try:
            time_series_preds, time_series_conf = time_series_prediction(data, prediction_days)
        except Exception as e:
            st.warning(f"Time series model failed: {e}. Using fallback.")
            # Simple fallback
            time_series_preds = [data['Close'].iloc[-1]] * prediction_days
            time_series_conf = [0.2] * prediction_days
            
        # Get ARIMA predictions directly for the ensemble as a fifth model
        try:
            arima_preds, arima_conf = arima_model_prediction(data, prediction_days)
        except Exception as e:
            st.warning(f"ARIMA model failed: {e}. Using fallback.")
            # Simple fallback
            arima_preds = [data['Close'].iloc[-1]] * prediction_days
            arima_conf = [0.2] * prediction_days
        
        # Check if all models are in the same scale by comparing means
        linear_mean = np.mean(linear_preds)
        quad_mean = np.mean(quad_preds)
        fourier_mean = np.mean(fourier_preds)
        ts_mean = np.mean(time_series_preds)
        arima_mean = np.mean(arima_preds)
        
        last_price = data['Close'].iloc[-1]
        
        # Convert outlier predictions back to reasonable scale if needed
        # (e.g., if a model predicts values 1000x higher than current price)
        scale_threshold = 5.0  # Flag predictions more than 5x the current price
        
        if linear_mean / last_price > scale_threshold:
            linear_preds = [p / (linear_mean / last_price) for p in linear_preds]
            linear_conf = [c * 0.5 for c in linear_conf]  # Reduce confidence in scaled predictions
            
        if quad_mean / last_price > scale_threshold:
            quad_preds = [p / (quad_mean / last_price) for p in quad_preds]
            quad_conf = [c * 0.5 for c in quad_conf]
            
        if fourier_mean / last_price > scale_threshold:
            fourier_preds = [p / (fourier_mean / last_price) for p in fourier_preds]
            fourier_conf = [c * 0.5 for c in fourier_conf]
            
        if ts_mean / last_price > scale_threshold:
            time_series_preds = [p / (ts_mean / last_price) for p in time_series_preds]
            time_series_conf = [c * 0.5 for c in time_series_conf]
            
        if arima_mean / last_price > scale_threshold:
            arima_preds = [p / (arima_mean / last_price) for p in arima_preds]
            arima_conf = [c * 0.5 for c in arima_conf]
        
        # Also ensure no negative prices
        linear_preds = [max(0, p) for p in linear_preds]
        quad_preds = [max(0, p) for p in quad_preds]
        fourier_preds = [max(0, p) for p in fourier_preds]
        time_series_preds = [max(0, p) for p in time_series_preds]
        arima_preds = [max(0, p) for p in arima_preds]
        
        # Combine all predictions with dynamic weighting
        model_weights = []
        predictions = []
        confidence = []
        
        # Initialize base regime weights
        base_weights = {
            'Linear Regression': regime_weights.get('Linear Regression', 0.2),
            'Quadratic Regression': regime_weights.get('Quadratic Regression', 0.2),
            'Fourier Transform': regime_weights.get('Fourier Transform', 0.2),
            'Time Series': regime_weights.get('Time Series', 0.2),
            'ARIMA': regime_weights.get('ARIMA', 0.2)
        }
        
        # Apply confidence-based adjustments for each day in the prediction horizon
        for i in range(prediction_days):
            # Get confidence scores for this prediction day
            day_confs = [
                linear_conf[i], 
                quad_conf[i], 
                fourier_conf[i], 
                time_series_conf[i],
                arima_conf[i]
            ]
            
            # Apply time horizon adjustment with regime awareness
            if current_regime == "Trending Up":
                # In uptrends, favor trend-following models for longer horizons
                if i < prediction_days // 3:
                    # Short-term: time series and ARIMA are most accurate
                    day_confs[3] *= 1.3  # time_series model
                    day_confs[4] *= 1.3  # ARIMA model
                elif i > (2 * prediction_days) // 3:
                    # Long-term: favor linear regression for extended uptrends
                    day_confs[0] *= 1.4  # linear model
            
            elif current_regime == "Trending Down":
                # In downtrends, favor models that can capture acceleration
                if i < prediction_days // 3:
                    # Short-term: ARIMA can capture recent momentum
                    day_confs[4] *= 1.4  # ARIMA model
                    day_confs[1] *= 1.2  # Quadratic model
                else:
                    # Medium to longer term: quadratic can capture accelerating downtrends
                    day_confs[1] *= 1.4  # Quadratic model
            
            elif current_regime == "Range-Bound":
                # In range-bound markets, favor oscillating models
                day_confs[2] *= 1.3  # Fourier model works better for oscillations
                day_confs[1] *= 1.2  # Quadratic can capture local curves
            
            # Create baseline weights from regime recommendations
            base_day_weights = [
                base_weights['Linear Regression'],
                base_weights['Quadratic Regression'],
                base_weights['Fourier Transform'],
                base_weights['Time Series'],
                base_weights['ARIMA']
            ]
            
            # Blend regime-based weights with confidence-based weights
            # 70% regime weights, 30% confidence weights
            regime_confidence = regime_info['confidence']
            if regime_confidence > 0.7:
                # If we're confident in the regime, use more regime-based weights
                regime_weight_factor = 0.7
            else:
                # If regime is less clear, rely more on model confidence
                regime_weight_factor = 0.4
                
            # Normalize confidence weights
            total_conf = sum(day_confs)
            if total_conf > 0:
                conf_weights = [conf / total_conf for conf in day_confs]
            else:
                conf_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
                
            # Blend weights
            weights = []
            for j in range(len(base_day_weights)):
                blended_weight = (base_day_weights[j] * regime_weight_factor) + (conf_weights[j] * (1 - regime_weight_factor))
                weights.append(blended_weight)
                
            # Renormalize blended weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Calculate weighted prediction
            day_preds = [
                linear_preds[i], 
                quad_preds[i], 
                fourier_preds[i], 
                time_series_preds[i],
                arima_preds[i]
            ]
            
            weighted_pred = sum(p * w for p, w in zip(day_preds, weights))
            
            # Store results
            predictions.append(weighted_pred)
            
            # Overall confidence is weighted by both model confidence and regime confidence
            model_confidence = max(day_confs)
            overall_confidence = (model_confidence * 0.7) + (regime_info['confidence'] * 0.3)
            confidence.append(overall_confidence)
            
            model_weights.append(weights)
        
        # Store the detected regime in the session state for reference
        if 'market_regime' not in st.session_state:
            st.session_state.market_regime = {}
            
        st.session_state.market_regime = {
            'regime': current_regime,
            'confidence': regime_info['confidence'],
            'regime_change': regime_info['regime_change'],
            'duration': regime_info['duration']
        }
        
        return predictions, confidence, model_weights
        
    except Exception as e:
        st.warning(f"Ensemble prediction failed: {e}. Using simple average.")
        
        # Simplified ensemble as a fallback
        try:
            linear_preds, _ = linear_regression_prediction(data, prediction_days)
            time_series_preds, _ = time_series_prediction(data, prediction_days)
            
            # Simple average of available predictions
            predictions = [(a + b) / 2 for a, b in zip(linear_preds, time_series_preds)]
            confidence = [0.4 * np.exp(-0.05 * i) for i in range(prediction_days)]
            
            # Simple equal weights for the two models
            model_weights = [[0.5, 0, 0, 0.5, 0] for _ in range(prediction_days)]
            
            # Set a default regime in session state
            if 'market_regime' not in st.session_state:
                st.session_state.market_regime = {}
                
            st.session_state.market_regime = {
                'regime': "Unknown",
                'confidence': 0.0,
                'regime_change': False,
                'duration': 0
            }
            
            return predictions, confidence, model_weights
            
        except:
            # Last resort fallback - simple trend extrapolation
            prices = data['Close'].values
            last_price = prices[-1]
            
            # Calculate average trend
            if len(prices) >= 10:
                avg_change = np.mean(np.diff(prices[-10:]))
            else:
                avg_change = 0
                
            # Linear extrapolation with exponential decay
            predictions = [max(0, last_price + avg_change * i * np.exp(-0.1 * i)) for i in range(1, prediction_days + 1)]
            confidence = [0.3 * np.exp(-0.05 * i) for i in range(prediction_days)]
            
            # Mock weights for UI consistency
            model_weights = [[0.2, 0.2, 0.2, 0.2, 0.2] for _ in range(prediction_days)]
            
            # Set a default regime in session state
            if 'market_regime' not in st.session_state:
                st.session_state.market_regime = {}
                
            st.session_state.market_regime = {
                'regime': "Unknown",
                'confidence': 0.0,
                'regime_change': False,
                'duration': 0
            }
            
            return predictions, confidence, model_weights

def plot_predictions(data, predictions, confidence, prediction_days=30):
    """
    Create a plot showing historical data and predictions.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
        predictions (list): List of predicted values
        confidence (list): List of confidence values
        prediction_days (int): Number of days predicted
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with predictions
    """
    # Create date range for predictions
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
    
    # Create figure
    fig = go.Figure()
    
    # Add historical price
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        )
    )
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            name='Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Add confidence interval
    upper_bound = [pred + pred * (1 - conf) * 0.2 for pred, conf in zip(predictions, confidence)]
    lower_bound = [pred - pred * (1 - conf) * 0.2 for pred, conf in zip(predictions, confidence)]
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper_bound,
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=lower_bound,
            mode='lines',
            name='Lower Bound',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        height=600
    )
    
    return fig

def plot_ensemble_weights(model_weights, prediction_days=30):
    """
    Create a plot showing model weights in the ensemble.
    
    Args:
        model_weights (list): List of weight distributions
        prediction_days (int): Number of days predicted
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with model weights
    """
    try:
        # Check if we have 5 weights per day (includes ARIMA model)
        using_arima = len(model_weights[0]) >= 5 if model_weights else False
        
        # Create arrays for each model's weights
        linear_weights = [weights[0] for weights in model_weights]
        quadratic_weights = [weights[1] for weights in model_weights]
        fourier_weights = [weights[2] for weights in model_weights]
        time_series_weights = [weights[3] for weights in model_weights]
        
        if using_arima:
            arima_weights = [weights[4] for weights in model_weights]
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each model's weights
        days = list(range(1, prediction_days + 1))
        
        fig.add_trace(
            go.Scatter(
                x=days,
                y=linear_weights,
                mode='lines+markers',
                name='Linear Regression',
                line=dict(width=2, color='blue')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=days,
                y=quadratic_weights,
                mode='lines+markers',
                name='Quadratic Regression',
                line=dict(width=2, color='red')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=days,
                y=fourier_weights,
                mode='lines+markers',
                name='Fourier Transform',
                line=dict(width=2, color='green')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=days,
                y=time_series_weights,
                mode='lines+markers',
                name='Time Series',
                line=dict(width=2, color='purple')
            )
        )
        
        if using_arima:
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=arima_weights,
                    mode='lines+markers',
                    name='ARIMA',
                    line=dict(width=2, color='orange')
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Model Weights in Ensemble Prediction',
            xaxis_title='Prediction Day',
            yaxis_title='Weight',
            template='plotly_white',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        # Fallback to a simpler visualization if there's an error
        st.warning(f"Error creating weight plot: {e}. Using simplified version.")
        
        # Create a simplified figure
        fig = go.Figure()
        
        # Add a single trace showing overall contribution
        days = list(range(1, prediction_days + 1))
        
        # Create dummy data if needed
        if not model_weights or len(model_weights) < prediction_days:
            dummy_weights = [[0.2, 0.2, 0.2, 0.2, 0.2] for _ in range(prediction_days)]
            model_names = ['Linear', 'Quadratic', 'Fourier', 'Time Series', 'ARIMA']
            
            for i, name in enumerate(model_names):
                fig.add_trace(
                    go.Scatter(
                        x=days,
                        y=[w[i] if i < len(w) else 0.2 for w in dummy_weights],
                        mode='lines',
                        name=name
                    )
                )
        else:
            # Use whatever data we have
            model_count = len(model_weights[0])
            model_names = ['Model ' + str(i+1) for i in range(model_count)]
            
            for i, name in enumerate(model_names):
                if i < model_count:
                    fig.add_trace(
                        go.Scatter(
                            x=days,
                            y=[w[i] if i < len(w) else 0 for w in model_weights],
                            mode='lines',
                            name=name
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title='Model Weights (Simplified View)',
            xaxis_title='Prediction Day',
            yaxis_title='Weight',
            template='plotly_white',
            height=400
        )
        
        return fig
