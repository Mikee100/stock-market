import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Modify your data loading section:
def load_data(asset, timeframe, period):
    try:
        data = yf.download(asset, period=period, interval=timeframe)
        # Filter out extreme historical periods (like COVID)
        if '2020' in period:  # If loading COVID period data
            data = data[data['Close'] > 1.00]  # Remove EURUSD < 1.00
        return data
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

def preprocess_data(data):
    """Clean and normalize data"""
    if data is None or len(data) < 30:
        return None, None
    
    try:
        data = data[['Close']].copy()
        data['Returns'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        
        scaler = MinMaxScaler()
        data[['Close_scaled']] = scaler.fit_transform(data[['Close']])
        return data, scaler
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, None

def create_features(data):
    """Create technical indicators"""
    if data is None or len(data) < 100:
        return None
    
    try:
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_30'] = data['Close'].rolling(window=30).mean()
        data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)
        data['Volatility_75'] = data['Returns'].rolling(window=75).std() * np.sqrt(252)
        data['Volatility_100'] = data['Returns'].rolling(window=100).std() * np.sqrt(252)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Feature creation error: {e}")
        return None

def create_sequences(data, seq_length):
    """Create time series sequences"""
    if data is None or len(data) < seq_length * 2:
        return None, None
    
    try:
        sequences = []
        targets = []
        for i in range(len(data)-seq_length-1):
            sequences.append(data[i:(i+seq_length)])
            targets.append(data[i+seq_length])
        
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32).reshape(-1, 1)
    except Exception as e:
        print(f"Sequence creation error: {e}")
        return None, None