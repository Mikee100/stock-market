import pandas as pd
import plotly.express as px
from transformers import pipeline
from datetime import datetime, timedelta
import requests
import numpy as np

class NewsSentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.news_api_key = "ee2b451947ac4ba897c3d4660ea2d5c9"  
        
    def fetch_financial_news(self, query="stock market", days=7):
        """Fetch recent financial news"""
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&sortBy=publishedAt&apiKey={self.news_api_key}"
        
        try:
            response = requests.get(url)
            articles = response.json().get('articles', [])
            return pd.DataFrame([{
                'title': a['title'],
                'date': pd.to_datetime(a['publishedAt']).date(),
                'source': a['source']['name'],
                'content': a['description']
            } for a in articles])
        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, news_df):
        """Analyze sentiment of news headlines"""
        if news_df.empty:
            return news_df
            
        # Analyze sentiment in batches
        texts = news_df['title'].fillna('') + " " + news_df['content'].fillna('')
        results = []
        batch_size = 10
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size].tolist()
            results.extend(self.sentiment_pipeline(batch))
            
        # Add sentiment to dataframe
        news_df['sentiment'] = [1 if r['label'] == 'POSITIVE' else -1 for r in results]
        news_df['sentiment_score'] = [r['score'] for r in results]
        
        return news_df
    
    def plot_sentiment_timeline(self, news_df):
        """Plot sentiment over time"""
        if news_df.empty:
            return None
            
        # Daily average sentiment
        daily_sentiment = news_df.groupby('date').agg({
            'sentiment': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'article_count'})
        
        fig = px.line(daily_sentiment, x=daily_sentiment.index, y='sentiment',
                     title="Financial News Sentiment Timeline",
                     labels={'sentiment': 'Sentiment Score', 'date': 'Date'},
                     hover_data=['article_count'])
        
        fig.update_layout(
            yaxis_range=[-1,1],
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Sentiment (Positive/Negative)",
            showlegend=False
        )
        
        # Add zero line and shaded areas
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_hrect(y0=0, y1=1, fillcolor="green", opacity=0.1)
        fig.add_hrect(y0=-1, y1=0, fillcolor="red", opacity=0.1)
        
        return fig