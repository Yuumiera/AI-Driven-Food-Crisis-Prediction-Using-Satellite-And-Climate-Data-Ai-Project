import requests
from textblob import TextBlob
import re
from datetime import datetime, timedelta

# News API configuration
NEWS_API_KEY = '9c0199db22344707914c11329b9bf06a'
NEWS_API_URL = 'https://newsapi.org/v2/everything'

def fetch_news():
    """
    Fetch news articles from News API with specific date range and relevance
    """
    # Get news from last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    params = {
        'q': '(food crisis OR climate crisis) AND (impact OR effect OR threat OR risk OR solution)',
        'language': 'en',
        'sortBy': 'relevancy',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'apiKey': NEWS_API_KEY
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        return response.json()['articles']
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def analyze_news(articles):
    """
    Analyze news articles and provide a concise summary with sentiment analysis
    """
    food_crisis_count = 0
    climate_crisis_count = 0
    total_articles = len(articles)
    
    if total_articles == 0:
        return "No news articles found to analyze."
    
    # Define comprehensive keyword categories with more specific terms
    food_crisis_categories = {
        'emergency': [
            'famine', 'starvation', 'severe food shortage',
            'acute food insecurity', 'food emergency'
        ],
        'global_threat': [
            'global food crisis', 'world food crisis',
            'food system collapse', 'food chain disruption'
        ],
        'humanitarian': [
            'humanitarian food crisis', 'emergency food aid',
            'food rationing', 'food vouchers'
        ],
        'production_crisis': [
            'crop failure', 'agricultural collapse',
            'harvest failure', 'food supply chain breakdown'
        ],
        'health_impact': [
            'hunger-related deaths', 'severe malnutrition',
            'acute undernourishment'
        ]
    }
    
    climate_crisis_categories = {
        'environmental': [
            'climate crisis', 'climate emergency',
            'climate breakdown', 'climate catastrophe'
        ],
        'natural_disasters': [
            'extreme drought', 'severe flooding',
            'devastating wildfire', 'catastrophic weather'
        ],
        'impact': [
            'climate disaster', 'climate catastrophe',
            'climate emergency', 'climate threat'
        ],
        'solutions': [
            'climate action', 'climate solution',
            'climate policy', 'climate goals'
        ],
        'scientific': [
            'climate science', 'climate data',
            'climate research', 'climate study'
        ]
    }
    
    # Initialize category counters and sentiment trackers
    food_category_counts = {category: 0 for category in food_crisis_categories}
    climate_category_counts = {category: 0 for category in climate_crisis_categories}
    food_sentiment_sum = 0
    climate_sentiment_sum = 0
    
    for article in articles:
        title = article['title']
        description = article['description']
        content = article.get('content', '')
        
        # Combine all text for analysis
        text = f"{title} {description} {content}"
        text = text.lower()
        
        # Get sentiment score
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        
        # Analyze food crisis categories
        for category, keywords in food_crisis_categories.items():
            if any(keyword in text for keyword in keywords):
                food_category_counts[category] += 1
                food_crisis_count += 1
                food_sentiment_sum += sentiment_score
                break
        
        # Analyze climate crisis categories
        for category, keywords in climate_crisis_categories.items():
            if any(keyword in text for keyword in keywords):
                climate_category_counts[category] += 1
                climate_crisis_count += 1
                climate_sentiment_sum += sentiment_score
                break
    
    # Calculate average sentiments
    food_avg_sentiment = food_sentiment_sum / food_crisis_count if food_crisis_count > 0 else 0
    climate_avg_sentiment = climate_sentiment_sum / climate_crisis_count if climate_crisis_count > 0 else 0
    
    # Generate concise summary
    summary = f"Analysis Results:\n"
    summary += f"• Total articles analyzed: {total_articles}\n"
    summary += f"• Food crisis related articles: {food_crisis_count} ({food_crisis_count/total_articles*100:.1f}% of total)\n"
    summary += f"• Climate crisis related articles: {climate_crisis_count} ({climate_crisis_count/total_articles*100:.1f}% of total)\n\n"
    
    if food_crisis_count == 0 and climate_crisis_count == 0:
        summary += "Neither food nor climate crisis issues are currently receiving significant media attention."
    else:
        # Determine severity based on category distribution
        food_emergency_ratio = (food_category_counts['emergency'] + food_category_counts['global_threat']) / food_crisis_count if food_crisis_count > 0 else 0
        climate_impact_ratio = (climate_category_counts['impact'] + climate_category_counts['natural_disasters']) / climate_crisis_count if climate_crisis_count > 0 else 0
        
        # Generate summary based on coverage, severity, and sentiment
        if food_crisis_count > 0:
            food_severity = "serious" if food_emergency_ratio > 0.3 else "moderate"
            food_sentiment = "negative" if food_avg_sentiment < -0.1 else "positive" if food_avg_sentiment > 0.1 else "neutral"
            summary += f"Food crisis issues are receiving {food_severity} media attention with a generally {food_sentiment} tone"
        
        if food_crisis_count > 0 and climate_crisis_count > 0:
            summary += " and "
        
        if climate_crisis_count > 0:
            climate_severity = "serious" if climate_impact_ratio > 0.3 else "moderate"
            climate_sentiment = "negative" if climate_avg_sentiment < -0.1 else "positive" if climate_avg_sentiment > 0.1 else "neutral"
            summary += f"climate crisis issues are receiving {climate_severity} media attention with a generally {climate_sentiment} tone"
        
        summary += "."
    
    return summary

def main():
    # Fetch news articles
    articles = fetch_news()
    
    if not articles:
        print("No articles found or error occurred while fetching news.")
        return
    
    # Analyze articles and get summary
    summary = analyze_news(articles)
    
    # Print summary
    print("\nNews Analysis Summary:")
    print("-" * 50)
    print(summary)
    print("-" * 50)

if __name__ == "__main__":
    main() 