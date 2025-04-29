import streamlit as st

st.set_page_config(page_title="Stock Market AI Assistant", layout="wide")

st.title("Stock Market AI Assistant")

st.write(
    """
    Use the sidebar to navigate between pages:
    - AI Trading Assistant
    - Market Trend Analysis
    - News Sentiment Analysis
    - Email Alert Setup
    """
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["AI Trading Assistant", "Market Trend Analysis", "News Sentiment Analysis", "Email Alert Setup"])

if page == "AI Trading Assistant":
    import pages.ai_trading_assistant as ai_trading
    ai_trading.app()
elif page == "Market Trend Analysis":
    import pages.market_trend_analysis as trend_analysis
    trend_analysis.app()
elif page == "News Sentiment Analysis":
    import pages.news_sentiment_analysis as news_sentiment
    news_sentiment.app()
elif page == "Email Alert Setup":
    import pages.email_alert_setup as email_alert
    email_alert.app()
