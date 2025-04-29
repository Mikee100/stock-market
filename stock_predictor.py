import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processor import load_data, preprocess_data, create_features, create_sequences
from model import build_lstm_model, train_model, predict_future
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
        
        # Build model
        model = build_lstm_model((seq_length, 1))
        
        # Train model
        with st.spinner(f"Training model for {n_epochs} epochs..."):
            model, history = train_model(model, X, y, epochs=n_epochs)
    
        with st.spinner("Making predictions..."):
            predictions = model.predict(X, verbose=0)
            predictions = scaler.inverse_transform(predictions)
            actual = scaler.inverse_transform(y)
            
            last_sequence = feature_data['Close_scaled'].values[-seq_length:]
            if future_hours is not None:
                future_predictions = predict_future(model, last_sequence, future_hours, scaler)
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
            else:
                future_predictions = predict_future(model, last_sequence, future_days, scaler)
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

# Add these additional parameters that were calculated separately
                trade_params.update({
                    'risk_percent': risk_percent,
                    'account_size': account_size,
                    'pip_value': 10 if asset == "EURUSD=X" else 1,
                    'pip_size': 0.0001 if asset == "EURUSD=X" else 0.1
                })

        display_trade_dashboard(trade_params)

        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(processed_data.index[seq_length:seq_length+len(actual)], actual, label='Actual Price')
        ax.plot(processed_data.index[seq_length:seq_length+len(predictions)], predictions, 
                label='Predicted Price', alpha=0.7)
        
        if future_hours is not None and len(future_dates_today) > 0:
            ax.plot(future_dates_today, future_predictions_today, 
                    label=f"Today's Next {len(future_dates_today)} Hours", 
                    linestyle='--', color='orange', marker='o')
        elif future_days is not None:
            ax.plot(future_dates, future_predictions, 
                    label=f'Next {future_days} Days Prediction', 
                    linestyle='--', color='orange')
        
        ax.set_title(f"{asset} Price Prediction")
        ax.legend()
        st.pyplot(fig) 

        
         # ============== TREND ANALYSIS SECTION ==============
        st.subheader("📊 Comprehensive Market Trend Analysis")
        with st.spinner("Analyzing broader market trends..."):
            with st.expander("Show Raw Data Stats"):
                 st.write(f"Data points: {len(data)}")
                 st.write("Columns:", data.columns.tolist())
                 st.write("Sample data:", data.head())
    
            trend_data, trend_fig = analyze_trend(data)
            
            if trend_fig is not None:
                st.pyplot(trend_fig)
                
                # Display trend summary metrics
                cols = st.columns(3)
                current_price = data['Close'].iloc[-1]
                price_fmt = f"{current_price:.4f}" if asset == "EURUSD=X" else f"{current_price:.2f}"
                cols[0].metric("Current Price", f"${price_fmt}")
                
                if 'Trend' in trend_data.columns:
                    trend_status = "Bullish" if trend_data['Trend'].iloc[-1] > 0 else "Bearish"
                    cols[1].metric("Market Trend", trend_status)
                
                if 'ADX' in trend_data.columns:
                    adx_value = trend_data['ADX'].iloc[-1]
                    adx_strength = "Strong" if adx_value > 25 else ("Weak" if adx_value < 20 else "Moderate")
                    cols[2].metric("Trend Strength", adx_strength)
                
                # Detailed view expander
                with st.expander("View Technical Indicators"):
                    display_cols = ['Close']
                    if 'SMA_50' in trend_data.columns:
                        display_cols.extend(['SMA_50', 'SMA_200'])
                    if 'MACD' in trend_data.columns:
                        display_cols.extend(['MACD', 'MACD_Signal'])
                    if 'ADX' in trend_data.columns:
                        display_cols.extend(['ADX'])
                    
                    st.dataframe(trend_data[display_cols].tail(10))
            else:
                st.warning("Basic trend analysis could not be generated. The data may be too limited.")
                st.info(f"Current data points: {len(data)} (Minimum recommended: 100)")
        # ============== END TREND ANALYSIS SECTION ============== 
        # ============== NEWS SENTIMENT SECTION ==============
        st.subheader("📰 Market News Sentiment")
    
    # Initialize analyzer
    news_analyzer = NewsSentimentAnalyzer()
    
    with st.expander("News Sentiment Analysis", expanded=True):
        # Get news data
        with st.spinner("Fetching and analyzing financial news..."):
            news_df = news_analyzer.fetch_financial_news(asset.split('=')[0])
            news_df = news_analyzer.analyze_sentiment(news_df)
        
        if not news_df.empty:
            # Show timeline
            fig = news_analyzer.plot_sentiment_timeline(news_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recent news with sentiment
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


        # ============== EMAIL ALERTS SECTION ==============
    
    # Email Alerts Section
    st.subheader("📧 Email Alert Setup")
    email_system = EmailAlertSystem()
    
    with st.expander("Email Configuration"):
        # Show current config (read-only)
        st.write("Current configuration:")
        st.json({
            "Sender": email_system.config['sender_email'],
            "Receiver": email_system.config['receiver_email'],
            "SMTP Server": email_system.config['smtp_server']
        })
        
        # Password update
        new_password = st.text_input(
            "Update Gmail App Password",
            type="password",
            help="Get app password from https://myaccount.google.com/apppasswords"
        )
        
        if st.button("Save Password"):
            if new_password:
                email_system.update_config(password=new_password)
                st.success("Password updated!")
            else:
                st.warning("Please enter a password")
    
    if st.button("Send Test Email"):
        if email_system.send_alert(
            "Test Alert from Trading System",
            "This is a test message from your trading bot."
        ):
            st.success("Test email sent successfully!")
        else:
            st.error("Failed to send test email. Check terminal for details.")
        
        
        # Display predictions
        if future_hours is not None and len(future_dates_today) > 0:
            st.subheader(f"Today's Next {len(future_dates_today)} Hour Prediction")
            fig_future, ax_future = plt.subplots(figsize=(10, 4))
            ax_future.plot(future_dates_today, future_predictions_today, 
                          marker='o', linestyle='-', color='red')
            ax_future.set_title(f"Today's Price Movement ({asset})")
            ax_future.set_xlabel("Time")
            ax_future.set_ylabel("Price")
            ax_future.grid(True)
            st.pyplot(fig_future)
            
            pred_df = pd.DataFrame({
                'Time': future_dates_today.strftime('%H:%M'),
                'Predicted Price': [f"${x:.2f}" for x in future_predictions_today.flatten()]
            })
            st.table(pred_df)
        elif future_days is not None:
            st.subheader(f"Next {future_days} Days Prediction")
            fig_future, ax_future = plt.subplots(figsize=(10, 4))
            ax_future.plot(future_dates, future_predictions, 
                          marker='o', linestyle='-', color='red')
            ax_future.set_title(f"Future Price Movement ({asset})")
            ax_future.set_xlabel("Date")
            ax_future.set_ylabel("Price")
            ax_future.grid(True)
            st.pyplot(fig_future)
        
        # Model performance
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Points", len(processed_data))
            st.metric("Sequence Length", seq_length)
        with col2:
            st.metric("Training Epochs", n_epochs)
            next_pred = future_predictions_today[0][0] if future_hours and len(future_dates_today) > 0 else future_predictions[0][0]
            st.metric("Next Prediction", f"${float(next_pred):.2f}")
        
        # Training loss
        st.subheader("Training Progress")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax_loss.plot(history.history['val_loss'], label='Validation Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        st.pyplot(fig_loss)

if __name__ == "__main__":
    main()