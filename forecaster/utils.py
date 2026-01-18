import pandas as pd
import numpy as np
import os
import requests
import joblib
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from django.conf import settings
import plotly.graph_objects as go
import time

# Configuration
TIME_STEP = 60
FORECAST_STEPS = 30
BASE_CURRENCY = "USD"
MODEL_PATH = settings.MODEL_ML_PATH 
CURRENCIES = [
    "EUR", "GBP", "USD", "CNY", "JPY", "CAD", "INR", "BRL", "RUB", "MXN",
    "AUD", "KRW", "TWD", "IDR", "TRY", "SAR", "CHF", "PLN", "ARS", "SEK",
    "SGD", "ILS", "AED", "THB", "NOK", "VND", "PHP", "BDT", "IRR", "DKK",
    "MYR", "COP", "HKD", "ZAR", "EGP", "RON", "PKR", "CZK", "CLP", "PEN", "KZT"
]
CURRENCIES.sort()

def add_indicators(df):
    """Consistent feature engineering for prediction and training."""
    df = df.copy()
    df['SMA_7'] = df['Rate'].rolling(window=7).mean()
    df['SMA_21'] = df['Rate'].rolling(window=21).mean()
    
    delta = df['Rate'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan) 
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.ffill().bfill()

def get_data(from_currency, to_currency):
    # Fetches data and handles API limits.
    load_dotenv(os.path.join(settings.BASE_DIR, '.env'))
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    URL = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&outputsize=full&apikey={API_KEY}"
    try:
        response = requests.get(URL, timeout=30)
        data = response.json()
        
        if "Note" in data or "Information" in data:
            return None, "API Rate limit reached. Please wait a minute."
        if 'Time Series FX (Daily)' not in data: 
            return None, "API Error: Data not found."
        
        ts_data = data['Time Series FX (Daily)']
        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df = df.rename(columns={'4. close': 'Rate'})
        df.index = pd.to_datetime(df.index)
        df['Rate'] = pd.to_numeric(df['Rate'])
        
        df = df.sort_index(ascending=True).asfreq('B').ffill()
        df = add_indicators(df)
        return df, None
    except Exception as e: 
        return None, f"Connection error: {str(e)}"

def create_sequences(dataset, time_step=TIME_STEP):
    # Creates multivariate sequences [Sample, Time, Feature].
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), :])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    # Builds model compatible with (TIME_STEP, 4) input.
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def perform_rolling_forecast(model, scaler, df_full):
    # Performs multivariate rolling forecast with robust unscaling.
    try:
        # Setup Initial Window
        last_features = df_full[['Rate', 'SMA_7', 'SMA_21', 'RSI']].values[-TIME_STEP:]
        current_batch_scaled = scaler.transform(last_features)
        current_batch = current_batch_scaled.reshape(1, TIME_STEP, 4)

        future_predictions = []
        temp_df = df_full[['Rate']].copy()

        # Pre-calculate unscaling parameters for the 'Rate' column (Index 0)
        # scaler.scale_[0] is (1 / (max - min)), scaler.min_[0] is the shift
        # formula: unscaled = (scaled - min) / scale
        rate_scale = scaler.scale_[0]
        rate_min = scaler.min_[0]

        for _ in range(FORECAST_STEPS):
            # Predict
            next_pred_scaled = model.predict(current_batch, verbose=0)[0, 0]
            
            # Robust Unscaling for the Rate column only
            unscaled_rate = (next_pred_scaled - rate_min) / rate_scale
            
            # Update history using Business Days
            new_date = temp_df.index[-1] + pd.offsets.BusinessDay(1)
            new_row = pd.DataFrame({'Rate': [unscaled_rate]}, index=[new_date])
            temp_df = pd.concat([temp_df, new_row])
            
            # Recalculate indicators for the new features
            temp_df_with_ind = add_indicators(temp_df)
            latest_row = temp_df_with_ind[['Rate', 'SMA_7', 'SMA_21', 'RSI']].values[-1:]
            
            # Scale newest row and shift window
            latest_scaled = scaler.transform(latest_row)
            current_batch = np.append(current_batch[:, 1:, :], latest_scaled.reshape(1, 1, 4), axis=1)
            
            future_predictions.append(unscaled_rate)

        forecast_dates = pd.date_range(
            start=df_full.index[-1] + pd.offsets.BusinessDay(1), 
            periods=FORECAST_STEPS, 
            freq='B'
        )
        return pd.Series(future_predictions, index=forecast_dates, name="Forecast"), None

    except Exception as e:
        return None, f"Rolling forecast failed: {str(e)}"

def generate_plotly_html(hist_data, forecast_data, from_currency, to_currency):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data.values, name='History', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data.values, name='Trend Forecast', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title=f'{from_currency}/{to_currency} Forecast (Price + Trend Indicators)',
        template='plotly_white',
        xaxis_title="Date",
        yaxis_title="Exchange Rate",
        hovermode="x unified"
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def load_forecasting_tools(from_currency, to_currency):
    model_file = os.path.join(MODEL_PATH, f'{from_currency}_{to_currency}_model.h5')
    scaler_file = os.path.join(MODEL_PATH, f'{from_currency}_{to_currency}_scaler.joblib')
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        return None, None, f"Model files for {from_currency}/{to_currency} not found."
    try:
        model = load_model(model_file)
        scaler = joblib.load(scaler_file)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

def build_and_train_live(df_full):
    """Builds and trains live. Note: Scaler fits on the FULL history for accuracy."""
    try:
        # Features are Rate, SMA_7, SMA_21, RSI
        data = df_full[['Rate', 'SMA_7', 'SMA_21', 'RSI']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Important: Fit on all historical data available to get the best min/max boundaries
        scaled_data = scaler.fit_transform(data)

        X_train, y_train = create_sequences(scaled_data, TIME_STEP)
        if len(X_train) == 0: return None, None, "Insufficient data sequences."

        model = build_model(input_shape=(TIME_STEP, 4))
        # Low epochs for live training to avoid web timeout
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


        
    