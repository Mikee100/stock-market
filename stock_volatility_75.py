import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_processor import load_data, preprocess_data, create_features, create_sequences
from model import build_lstm_model, train_model, predict_future

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("Stock Market Volatility 75 Predictor")
st.subheader("Volatility (75-day) Prediction with LSTM")

def main():
    st.sidebar.header("Settings")
    asset = st.sidebar.selectbox("Select Asset", ["EURUSD=X", "GC=F"])
    timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Hourly", "10-Minute"])
    
    if timeframe == "Daily":
        years = st.sidebar.slider("Years of Historical Data", 1, 10, 5)
        period = f"{years}y"
        interval = '1d'
        future_steps_label = "Future Days to Predict"
        future_steps_max = 30
        future_steps_default = 3
        freq = 'D'
        future_steps = st.sidebar.slider(future_steps_label, 1, future_steps_max, future_steps_default)
    elif timeframe == "Hourly":
        days = st.sidebar.slider("Days of Historical Data (max 60)", 1, 60, 30)
        period = f"{days}d"
        interval = '60m'
        future_steps_label = "Future Hours to Predict"
        future_steps_max = 48
        future_steps_default = 48
        freq = 'H'
        future_steps = st.sidebar.slider(future_steps_label, 1, future_steps_max, future_steps_default)
    else:  # 10-Minute interval
        days = st.sidebar.slider("Days of Historical Data (max 1)", 1, 1, 1)
        period = f"{days}d"
        interval = '10m'
        future_steps_label = "Future 10-Minute Intervals to Predict"
        future_steps_max = 6 * 24  # 6 intervals per hour * 24 hours = 144
        future_steps_default = 144
        freq = '10T'
        future_steps = st.sidebar.slider(future_steps_label, 1, future_steps_max, future_steps_default)
    
    seq_length = st.sidebar.slider("Sequence Length", 30, 90, 60)
    n_epochs = st.sidebar.slider("Number of Epochs", 10, 100, 30)
    
    # Load data
    data = load_data(asset, timeframe=interval, period=period)
    if data is None:
        st.error("Failed to load data. Please try different settings.")
        return
    
    with st.spinner("Processing data..."):
        processed_data, scaler = preprocess_data(data)
        if processed_data is None or scaler is None:
            st.error("Data preprocessing failed. Please try different settings.")
            return
        
        # Use volatility_75 feature instead of Close_scaled
        feature_data = create_features(processed_data)
        if feature_data is None or 'Volatility_75' not in feature_data.columns:
            st.error("Feature creation failed or Volatility_75 not found. Please try different settings.")
            return
        
        X, y = create_sequences(feature_data['Volatility_75'].values, seq_length)
        if X is None or y is None:
            st.error("Sequence creation failed. Please try different settings.")
            return
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        model = build_lstm_model((seq_length, 1))
        
        with st.spinner(f"Training model for {n_epochs} epochs..."):
            model, history = train_model(model, X, y, epochs=n_epochs)
        
        with st.spinner("Making predictions..."):
            predictions = model.predict(X, verbose=0)
            # For volatility, scaling might be different; inverse transform if scaler supports it
            try:
                predictions = scaler.inverse_transform(predictions)
                actual = scaler.inverse_transform(y)
            except Exception:
                # If inverse_transform fails, use raw values
                actual = y
            last_sequence = feature_data['Volatility_75'].values[-seq_length:]
            future_predictions = predict_future(model, last_sequence, future_steps, scaler)
        
        # Plot results with more detail
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(processed_data.index[seq_length:seq_length+len(actual)], actual, label='Actual Volatility 75', marker='o', markersize=3)
        ax.plot(processed_data.index[seq_length:seq_length+len(predictions)], predictions, label='Predicted Volatility 75', alpha=0.7, marker='x', markersize=3)
        
        future_dates = pd.date_range(start=processed_data.index[-1] + pd.Timedelta('1' + freq), periods=future_steps, freq=freq)
        ax.plot(future_dates, future_predictions, label=f'Future Volatility 75 Predictions (Next {future_steps} intervals)', linestyle='--', color='orange', marker='^', markersize=4)
        
        ax.set_title(f"{asset} Volatility 75 Prediction")
        ax.legend()
        ax.grid(True)
        
        # Improve x-axis formatting for dates
        if freq == 'D':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, future_steps//10)))
        elif freq == 'H':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, future_steps//10)))
        else:  # 10T
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        
        fig.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)
        
        # Separate chart showing only future predicted movements with more detail
        st.subheader("Future Volatility 75 Movements")
        fig_future, ax_future = plt.subplots(figsize=(10, 4))
        ax_future.plot(future_dates, future_predictions, marker='o', linestyle='-', color='red')
        ax_future.set_title("Future Volatility 75 Prediction")
        ax_future.set_xlabel("Date/Time")
        ax_future.set_ylabel("Predicted Volatility 75")
        ax_future.grid(True)
        ax_future.legend(["Future Prediction"])
        
        if freq == 'D':
            ax_future.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_future.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, future_steps//10)))
        elif freq == 'H':
            ax_future.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax_future.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, future_steps//10)))
        else:
            ax_future.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax_future.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        
        fig_future.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig_future)
        
        # Show model performance and training metrics
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Data Points", len(processed_data))
            st.metric("Sequence Length Used", seq_length)
        with col2:
            st.metric("Training Epochs", n_epochs)
            st.metric("Latest Prediction", f"{future_predictions[-1][0]:.4f}")
            st.metric("Future Prediction", f"{future_predictions[-1][0]:.4f}")
        
        # Plot training loss
        st.subheader("Training Loss")
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
