import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def calculate_trade_parameters(asset, account_size, risk_percent, 
                             current_price, predicted_prices, 
                             processed_data, future_hours, future_days, 
                             timeframe, future_dates_today=None, 
                             future_predictions_today=None):
    """
    Calculate all trade execution parameters
    Returns a dictionary with all calculated values
    """
    # Set price formatting based on asset
    price_format = ".4f" if asset == "EURUSD=X" else ".2f"
    
    # Calculate price movement
    price_change = predicted_prices[-1] - current_price
    pct_change = (price_change / current_price) * 100
    
    # Trading Strategy Logic
    if len(predicted_prices) > 1:
        # Calculate trend metrics
        price_velocity = (predicted_prices[-1] - predicted_prices[0]) / len(predicted_prices)
        price_acceleration = (predicted_prices[-1] - 2*predicted_prices[len(predicted_prices)//2] + predicted_prices[0]) / (len(predicted_prices)**2)
        
        # Enhanced entry decision with momentum confirmation
        if price_velocity > 0 and price_acceleration > 0:
            action = "BUY"
            confidence = "High Momentum"
            stop_loss_pct = 0.004
            take_profit_multiplier = 2.0
        elif price_velocity > 0:
            action = "BUY"
            confidence = "Steady Trend"
            stop_loss_pct = 0.006
            take_profit_multiplier = 1.5
        elif price_velocity < 0 and price_acceleration < 0:
            action = "SELL"
            confidence = "Strong Downtrend"
            stop_loss_pct = 0.004
            take_profit_multiplier = 2.0
        elif price_velocity < 0:
            action = "SELL"
            confidence = "Steady Decline"
            stop_loss_pct = 0.006
            take_profit_multiplier = 1.5
        else:
            action = "HOLD"
            confidence = "No Clear Trend"
    else:
        action = "HOLD"
        confidence = "Insufficient Data"

    # Calculate trade parameters if not HOLD
    if action in ["BUY", "SELL"]:
        entry_point = current_price
        
        if action == "BUY":
            stop_loss = entry_point * (1 - stop_loss_pct)
            take_profit = entry_point + take_profit_multiplier * (entry_point - stop_loss)
        else:  # SELL
            stop_loss = entry_point * (1 + stop_loss_pct)
            take_profit = entry_point - take_profit_multiplier * (stop_loss - entry_point)
        
        risk_reward = take_profit_multiplier
        
        # Position sizing calculation
        risk_amount = account_size * (risk_percent / 100)
        
        if asset == "EURUSD=X":
            pip_value = 10  # $10 per pip for standard lot
            pip_size = 0.0001
        else:  # GC=F
            pip_value = 1  # $1 per pip for standard lot
            pip_size = 0.1
        
        pip_risk = abs(entry_point - stop_loss) / pip_size
        lot_size = round((risk_amount / (pip_risk * pip_value)), 2)
        
        # Calculate position value
        position_value = lot_size * (100000 if asset == "EURUSD=X" else 100) * entry_point
        
        # Trade duration estimate
        duration = f"{future_hours} hours" if future_hours else f"{future_days} days"
        
        # Trade quality score (0-100)
        trade_score = min(100, max(0, 60 + (40 * (abs(pct_change)/2))))
        
        # Trade probability estimation
        win_probability = min(90, max(55, 60 + (30 * (abs(pct_change)/2))))
    else:
        entry_point = take_profit = stop_loss = risk_reward = lot_size = position_value = None
        trade_score = win_probability = 0

    return {
        'action': action,
        'confidence': confidence,
        'current_price': current_price,
        'price_format': price_format,
        'pct_change': pct_change,
        'entry_point': entry_point,
        'stop_loss': stop_loss,
        'stop_loss_pct': stop_loss_pct,
        'take_profit': take_profit,
        'risk_reward': risk_reward,
        'lot_size': lot_size,
        'position_value': position_value,
        'risk_amount': risk_amount,
        'pip_risk': pip_risk,
        'duration': duration,
        'trade_score': trade_score,
        'win_probability': win_probability,
        'processed_data': processed_data,
        'future_hours': future_hours,
        'future_days': future_days,
        'timeframe': timeframe,
        'future_dates_today': future_dates_today,
        'future_predictions_today': future_predictions_today,
        'asset': asset
    }

def display_trade_dashboard(params):
    """Display the trade execution dashboard"""
    st.subheader("🎯 Trade Execution Plan")
    
    # TRADE EXECUTION DASHBOARD
    st.markdown("### 📈 Trade Signal")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${params['current_price']:{params['price_format']}}")
        st.metric("Predicted Move", f"{params['pct_change']:.2f}%",
                 delta_color="inverse" if params['pct_change'] < 0 else "normal")
    
    with col2:
        st.metric("Action Signal", params['action'], help=f"Based on trend analysis: {params['confidence']}")
        st.metric("Confidence Score", f"{params['trade_score']:.0f}/100")
    
    with col3:
        if params['action'] in ["BUY", "SELL"]:
            st.metric("Win Probability", f"{params['win_probability']:.0f}%")
            st.metric("Risk/Reward", f"1:{params['risk_reward']:.1f}")

    # TRADE EXECUTION DETAILS
    if params['action'] in ["BUY", "SELL"]:
        st.success("✨ Strong Trading Opportunity Detected!")
        
        execution_col1, execution_col2 = st.columns([1, 2])
        
        with execution_col1:
            st.markdown("""
            ### 🎯 Execution Plan
            **Entry Price**: ${entry_price}  
            **Stop Loss**: ${stop_loss} ({stop_loss_pct:.1%} risk)  
            **Take Profit**: ${take_profit}  
            **Position Size**: {lot_size} lots (${position_value:,.2f})  
            **Risk Amount**: ${risk_amount:,.2f} ({risk_percent}% of account)  
            **Pips at Risk**: {pip_risk:.1f} pips  
            **Duration**: {duration}
            """.format(
                entry_price=f"{params['entry_point']:{params['price_format']}}",
                stop_loss=f"{params['stop_loss']:{params['price_format']}}",
                stop_loss_pct=params['stop_loss_pct'],
                take_profit=f"{params['take_profit']:{params['price_format']}}",
                lot_size=params['lot_size'],
                position_value=params['position_value'],
                risk_amount=params['risk_amount'],
                risk_percent=params['risk_percent'],
                pip_risk=params['pip_risk'],
                duration=params['duration']
            ))
            
            if st.button("📊 Show Trade Calculator"):
                st.session_state.show_calculator = not st.session_state.get('show_calculator', False)
            
            if st.session_state.get('show_calculator', False):
                st.markdown("### Position Size Calculator")
                new_risk = st.slider("Adjust Risk Percentage", 0.1, 5.0, float(params['risk_percent']), 0.1)
                new_lot_size = round((params['account_size'] * (new_risk/100) / (params['pip_risk'] * params['pip_value'])), 2)
                st.write(f"New Lot Size: {new_lot_size} lots (${new_lot_size * (100000 if params['asset'] == 'EURUSD=X' else 100) * params['entry_point']:,.2f})")
        
        with execution_col2:
            # Enhanced trade visualization
            fig_trade, ax_trade = plt.subplots(figsize=(10, 4))
            
            # Price history
            history_days = min(5, len(params['processed_data']))
            hist_prices = params['processed_data']['Close'].iloc[-history_days*24 if params['timeframe'] == "Hourly" else -history_days:]
            ax_trade.plot(hist_prices.index, hist_prices, label='Price History', color='blue')
            
            # Current price marker
            ax_trade.axhline(y=params['current_price'], color='gray', linestyle='--', alpha=0.5)
            ax_trade.scatter([params['processed_data'].index[-1]], [params['current_price']], color='black', s=100, label='Current Price')
            
            # Trade levels
            if params['action'] == "BUY":
                ax_trade.axhline(y=params['stop_loss'], color='red', linestyle='-', label='Stop Loss')
                ax_trade.axhline(y=params['take_profit'], color='green', linestyle='-', label='Take Profit')
            else:
                ax_trade.axhline(y=params['stop_loss'], color='red', linestyle='-', label='Stop Loss')
                ax_trade.axhline(y=params['take_profit'], color='green', linestyle='-', label='Take Profit')
            
            # Future predictions
            if params['future_hours'] and len(params['future_dates_today']) > 0:
                ax_trade.plot(params['future_dates_today'], params['future_predictions_today'], 
                            linestyle='--', color='orange', marker='o', label='Predicted Prices')
            
            ax_trade.set_title(f"{params['action']} Signal - {params['asset']}")
            ax_trade.set_xlabel("Time")
            ax_trade.set_ylabel("Price")
            ax_trade.legend()
            ax_trade.grid(True)
            st.pyplot(fig_trade)
    
    elif params['action'] == "HOLD":
        st.warning("""
        ⚠️ No Strong Trading Signal Detected  
        Recommendation: Wait for better market conditions with clearer trends
        """)