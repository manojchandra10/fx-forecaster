import pandas as pd
import numpy as np
import os
import requests
import joblib
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time

# Configuration
TIME_STEP = 60
BASE_CURRENCY = "USD"
CURRENCIES = [
    "EUR", "GBP", "USD", "CNY", "JPY", "CAD", "INR", "BRL", "RUB",
    "AUD", "TWD", "CHF", "PLN", "SEK", "SGD", "AED", "NOK", "DKK",
    "HKD", "ZAR", "PKR"
]
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')

def add_indicators(df):
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
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    URL = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&outputsize=full&apikey={API_KEY}"
    
    try:
        # Alpha Vantage Free Tier: 5 calls/min. 15s sleep is safe.
        time.sleep(15) 
        response = requests.get(URL)
        data = response.json()
        
        if 'Time Series FX (Daily)' not in data: 
            print(f"Error fetching {from_currency}: {data.get('Note', 'Unknown Error')}")
            return None
        
        ts_data = data['Time Series FX (Daily)']
        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df = df.rename(columns={'4. close': 'Rate'})
        df.index = pd.to_datetime(df.index)
        df['Rate'] = pd.to_numeric(df['Rate'])
        df = df.sort_index(ascending=True)
        # Use Business Day frequency
        df = df[['Rate']].asfreq('B').ffill()
        
        df = add_indicators(df)
        return df
    except Exception as e: 
        print(f"Exception for {from_currency}: {e}")
        return None

def create_sequences(dataset, time_step=TIME_STEP):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), :]) 
        y.append(dataset[i + time_step, 0])      
    return np.array(X), np.array(y)

def train_and_save_model(from_currency, to_currency):
    model_file = os.path.join(MODEL_PATH, f'{from_currency}_{to_currency}_model.h5')
    scaler_file = os.path.join(MODEL_PATH, f'{from_currency}_{to_currency}_scaler.joblib')
    
    df = get_data(from_currency, to_currency)
    if df is None: return

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    X_train, y_train = create_sequences(scaled_data, TIME_STEP)
    if len(X_train) < 100: 
        print(f"Skipping {from_currency}: Insufficient data.")
        return
        
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIME_STEP, 4)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    
    print(f"Training {from_currency}/{to_currency}...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.15, callbacks=[es], verbose=0)
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save(model_file)
    joblib.dump(scaler, scaler_file)
    print(f"--- Successfully Saved {from_currency}/{to_currency} ---")

if __name__ == "__main__":
    for currency in CURRENCIES:
        if currency != BASE_CURRENCY:
            train_and_save_model(currency, BASE_CURRENCY)