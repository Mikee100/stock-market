import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Tuple, Optional

class TrendAnalyzer:
    def __init__(self):
        self.min_data_points = 100  # Reduced minimum requirement
    
    def analyze_trend(self, data: pd.DataFrame, window: int = 21, polyorder: int = 2) -> Tuple[Optional[pd.DataFrame], Optional[plt.Figure]]:
        try:
            # Enhanced validation
            if not self._validate_input(data):
                return None, None
                
            df = self._prepare_data(data.copy())
            df = self._calculate_basic_indicators(df)
            
            # Only calculate ADX if we have enough data
            if len(df) >= 30:
                df['ADX'], df['+DI'], df['-DI'] = self._calculate_adx(df['High'], df['Low'], df['Close'])
                df['Trend_Strength'] = df['ADX'].apply(lambda x: 0 if x < 20 else (1 if x < 40 else 2))
            else:
                df['ADX'] = np.nan
                df['+DI'] = np.nan
                df['-DI'] = np.nan
                df['Trend_Strength'] = 0
            
            # Smooth price if enough data
            if len(df) > window:
                df['Smooth_Close'] = savgol_filter(df['Close'].values, 
                                                 window_length=min(window, len(df)-1), 
                                                 polyorder=min(polyorder, window-1))
            else:
                df['Smooth_Close'] = df['Close']
            
            fig = self._create_trend_plot(df)
            return df.dropna(), fig
            
        except Exception as e:
            print(f"Trend analysis error: {str(e)}")
            return None, None

    def _validate_input(self, data: pd.DataFrame) -> bool:
        if data is None or len(data) < 20:  # Absolute minimum
            return False
            
        required_cols = ['Close']
        if not all(col in data.columns for col in required_cols):
            return False
            
        # Ensure we have numeric data
        try:
            pd.to_numeric(data['Close'])
            return True
        except:
            return False
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data is clean and has required columns"""
        if 'High' not in df.columns:
            df['High'] = df['Close']
        if 'Low' not in df.columns:
            df['Low'] = df['Close']
        return df
    
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators that don't require much history"""
        # Simple moving averages
        df['SMA_50'] = df['Close'].rolling(min_periods=1, window=50).mean()
        df['SMA_200'] = df['Close'].rolling(min_periods=1, window=200).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Basic MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Simple trend detection
        df['Trend'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
        
        return df
    
    def _create_trend_plot(self, df: pd.DataFrame) -> plt.Figure:
        """Create visualization with available data"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Price and Trends
            ax.plot(df.index, df['Close'], label='Price', color='blue', alpha=0.7)
            
            if 'Smooth_Close' in df.columns:
                ax.plot(df.index, df['Smooth_Close'], label='Trend Line', color='red', linewidth=1.5)
            
            if 'SMA_50' in df.columns:
                ax.plot(df.index, df['SMA_50'], label='50 SMA', color='orange', linestyle='--')
            
            if 'SMA_200' in df.columns:
                ax.plot(df.index, df['SMA_200'], label='200 SMA', color='green', linestyle='--')
            
            ax.set_title('Price Trend Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Plot creation error: {str(e)}")
            return None

def analyze_trend(data, window=21, polyorder=2):
    return TrendAnalyzer().analyze_trend(data, window, polyorder)