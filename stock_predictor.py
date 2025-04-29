import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from data_processor import load_data, preprocess_data, create_features, create_sequences
from lstm_forecaster import LSTMForecaster
from trend_analyzer import analyze_trend
from email_alerts import EmailAlertSystem
from news_sentiment import NewsSentimentAnalyzer
from trade_execution import calculate_trade_parameters, display_trade_dashboard

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI Trading Assistant")
st.subheader("Smart Entry Points & Position Sizing Calculator")

def main():
    st.sidebar.header("Trading Parameters")
    asset = st.sidebar.selectbox("Select Asset", ["EURUSD=X", "GC=F"])
    timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Hourly"])
    
    # Risk Management Settings
    st.sidebar.header("Risk Management")
    account_size = st.sidebar.number_input("Account Balance ($)", 1000, 1000000, 10000)
    risk_percent = st.sidebar.slider("Risk Percentage per Trade", 0.1, 5.0, 1.0)
    
    if timeframe == "Daily":
        years = st.sidebar.slider("Years of Historical Data", 1, 10, 5)
        period = f"{years}y"
        interval = '1d'
        future_days = st.sidebar.slider("Future Days to Predict", 1, 30, 3)
        future_hours = None
    else:
        days = st.sidebar.slider("Days of Historical Data (max 60)", 1, 60, 30)
        period = f"{days}d"
        interval = '60m'
        future_hours = 2  # Fixed to predict next 2 hours
        future_days = None
        st.sidebar.write(f"Predicting next {future_hours} hours for today only")

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
        
        feature_data = create_features(processed_data)
        if feature_data is None:
            st.error("Feature creation failed. Please try different settings.")
            return
        
        X, y = create_sequences(feature_data['Close_scaled'].values, seq_length)
        if X is None or y is None:
            st.error("Sequence creation failed. Please try different settings.")
            return
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Initialize and train the enhanced LSTM model
        forecaster = LSTMForecaster(input_shape=(seq_length, 1), units=128)
        forecaster.scaler = scaler
        
        # Train model
        with st.spinner(f"Training model for {n_epochs} epochs..."):
            history = forecaster.train(X, y, epochs=n_epochs, batch_size=64)
    
        with st.spinner("Making predictions..."):
            # Get predictions for training data
            predictions = forecaster.model.predict(X, verbose=0)
            predictions = scaler.inverse_transform(predictions)
            actual = scaler.inverse_transform(y)
            
            # Get future predictions with uncertainty
            last_sequence = feature_data['Close_scaled'].values[-seq_length:]
            if future_hours is not None:
                future_predictions, pred_std = forecaster.predict_with_uncertainty(
                    last_sequence, 
                    n_steps=future_hours,
                    n_samples=50
                )
                future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1))
                pred_std = scaler.inverse_transform(pred_std.reshape(-1, 1))
                
                future_dates = pd.date_range(
                    start=processed_data.index[-1] + pd.Timedelta(minutes=60),
                    periods=future_hours,
                    freq='H'
                )
                # Filter to only show today's predictions
                today = pd.Timestamp.now().normalize()
                mask = future_dates.normalize() == today
                future_dates_today = future_dates[mask]
                future_predictions_today = future_predictions[mask]
                pred_std_today = pred_std[mask]
            else:
                future_predictions, pred_std = forecaster.predict_with_uncertainty(
                    last_sequence,
                    n_steps=future_days,
                    n_samples=50
                )
                future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1))
                pred_std = scaler.inverse_transform(pred_std.reshape(-1, 1))
                
                future_dates = pd.date_range(
                    start=processed_data.index[-1] + pd.Timedelta(days=1),
                    periods=future_days,
                    freq='B'
                )

        # TRADE EXECUTION SYSTEM
        trade_params = calculate_trade_parameters(
            asset=asset,
            account_size=account_size,
            risk_percent=risk_percent,
            current_price=float(processed_data['Close'].iloc[-1]),
            predicted_prices=[float(x[0]) for x in (future_predictions_today if future_hours else future_predictions)],
            processed_data=processed_data,
            future_hours=future_hours,
            future_days=future_days,
            timeframe=timeframe,
            future_dates_today=future_dates_today if future_hours else None,
            future_predictions_today=future_predictions_today if future_hours else None
        )

        trade_params.update({
            'risk_percent': risk_percent,
            'account_size': account_size,
            'pip_value': 10 if asset == "EURUSD=X" else 1,
            'pip_size': 0.0001 if asset == "EURUSD=X" else 0.1
        })

        display_trade_dashboard(trade_params)

        # Plot results with Plotly for interactive visualization
        st.subheader(f"{asset} Price Prediction with Confidence Intervals")
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=processed_data.index[seq_length:seq_length+len(actual)],
            y=actual.flatten(),
            name='Actual Price',
            line=dict(color='blue', width=2)
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=processed_data.index[seq_length:seq_length+len(predictions)],
            y=predictions.flatten(),
            name='Predicted Price',
            line=dict(color='green', width=2, dash='dot')
        ))
        
        # Replace the problematic plotting section with this:

        if future_hours is not None and len(future_dates_today) > 0:
                # Create confidence interval trace
                fig.add_trace(go.Scatter(
                    x=np.concatenate([future_dates_today.values, future_dates_today.values[::-1]]),
                    y=np.concatenate([
                        (future_predictions_today.flatten() + 1.96*pred_std_today.flatten()),
                        (future_predictions_today.flatten() - 1.96*pred_std_today.flatten())[::-1]
                    ]),
                    fill='toself',
                    fillcolor='rgba(255,165,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=future_dates_today,
                    y=future_predictions_today.flatten(),
                    name=f"Next {len(future_dates_today)} Hour Prediction",
                    line=dict(color='orange', width=2),
                    mode='lines+markers'
                ))
        elif future_days is not None:
                # Create confidence interval trace
                fig.add_trace(go.Scatter(
                    x=np.concatenate([future_dates.values, future_dates.values[::-1]]),
                    y=np.concatenate([
                        (future_predictions.flatten() + 1.96*pred_std.flatten()),
                        (future_predictions.flatten() - 1.96*pred_std.flatten())[::-1]
                    ]),
                    fill='toself',
                    fillcolor='rgba(255,165,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions.flatten(),
                    name=f"Next {future_days} Days Prediction",
                    line=dict(color='orange', width=2),
                    mode='lines+markers'
                ))
        
        fig.update_layout(
            title=f"{asset} Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(processed_data))
            st.metric("Sequence Length", seq_length)
        with col2:
            st.metric("Training Epochs", len(history.history['loss']))
            next_pred = future_predictions_today[0][0] if future_hours and len(future_dates_today) > 0 else future_predictions[0][0]
            st.metric("Next Prediction", f"${float(next_pred):.2f}")
        with col3:
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
            st.metric("Final Training Loss", f"{final_loss:.4f}")
            if final_val_loss:
                st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
        
        # Training progress
        st.subheader("Training Progress")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=history.history['loss'],
            name='Training Loss',
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        if 'val_loss' in history.history:
            fig_loss.add_trace(go.Scatter(
                y=history.history['val_loss'],
                name='Validation Loss',
                mode='lines',
                line=dict(color='red', width=2)
            ))
        fig_loss.update_layout(
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Future predictions table
      # In the visualization section where you create the Plotly figure, replace the problematic code with:

        if future_hours is not None and len(future_dates_today) > 0:
            # Convert DatetimeIndex to numpy array for concatenation
            dates_array = future_dates_today.to_numpy()
            fig.add_trace(go.Scatter(
                x=np.concatenate([dates_array, dates_array[::-1]]),
                y=np.concatenate([
                    (future_predictions_today.flatten() + 1.96*pred_std_today.flatten()),
                    (future_predictions_today.flatten() - 1.96*pred_std_today.flatten())[::-1]
                ]),
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
        elif future_days is not None:
            # Convert DatetimeIndex to numpy array for concatenation
            dates_array = future_dates.to_numpy()
            fig.add_trace(go.Scatter(
                x=np.concatenate([dates_array, dates_array[::-1]]),
                y=np.concatenate([
                    (future_predictions.flatten() + 1.96*pred_std.flatten()),
                    (future_predictions.flatten() - 1.96*pred_std.flatten())[::-1]
                ]),
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
        # Model evaluation metrics
        st.subheader("Model Evaluation Metrics")
        eval_metrics = forecaster.evaluate_model(X[-100:], y[-100:])  # Evaluate on last 100 samples
        st.json(eval_metrics)
        
        # Save model option
        if st.button("💾 Save Model"):
            forecaster.save_model(f"{asset.replace('=X','')}_{timeframe}_model.h5")
            st.success("Model saved successfully!")

    # News Sentiment Analysis
    st.subheader("📰 Market News Sentiment")
    news_analyzer = NewsSentimentAnalyzer()
    
    with st.expander("News Sentiment Analysis", expanded=True):
        with st.spinner("Fetching and analyzing financial news..."):
            news_df = news_analyzer.fetch_financial_news(asset.split('=')[0])
            news_df = news_analyzer.analyze_sentiment(news_df)
        
        if not news_df.empty:
            fig = news_analyzer.plot_sentiment_timeline(news_df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("### Recent News Headlines")
            cols = st.columns([1,3,1,1])
            cols[0].write("**Date**")
            cols[1].write("**Headline**")
            cols[2].write("**Sentiment**")
            cols[3].write("**Score**")
            
            for _, row in news_df.sort_values('date', ascending=False).head(10).iterrows():
                cols = st.columns([1,3,1,1])
                cols[0].write(row['date'].strftime('%b %d'))
                cols[1].write(row['title'])
                sentiment_icon = "👍" if row['sentiment'] > 0 else "👎"
                cols[2].write(f"{sentiment_icon} {'Positive' if row['sentiment'] > 0 else 'Negative'}")
                cols[3].write(f"{row['sentiment_score']:.0%}")
        else:
            st.warning("Could not fetch news data. Please check your API key and internet connection.")

if __name__ == "__main__":
    main()