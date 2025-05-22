# import requests
# from textblob import TextBlob
# import re
# from datetime import datetime, timedelta
# import os

# # News API configuration
# NEWS_API_KEY = '9c0199db22344707914c11329b9bf06a'
# NEWS_API_URL = 'https://newsapi.org/v2/everything'

# def save_analysis_to_file(analysis_result, country=None):
#     """
#     Save analysis results to a text file
#     """
#     # Get the absolute path of the project root directory
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
#     # Create results directory if it doesn't exist
#     results_dir = os.path.join(project_root, 'results')
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
    
#     # Create filename with timestamp
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f"news_analysis_{country}_{timestamp}.txt" if country else f"news_analysis_{timestamp}.txt"
#     filepath = os.path.join(results_dir, filename)
    
#     # Save analysis to file
#     with open(filepath, 'w', encoding='utf-8') as f:
#         f.write(analysis_result)
    
#     print(f"Analysis saved to: {filepath}")  # Debug print
#     return filepath

# def fetch_news(country=None):
#     """
#     Fetch news articles from News API with specific date range and relevance
#     """
#     # Get news from last 7 days
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=7)
    
#     # Base query for food and climate crisis
#     base_query = '(food crisis OR climate crisis) AND (impact OR effect OR threat OR risk OR solution)'
    
#     # Add country to query if specified
#     if country:
#         query = f'({base_query}) AND ({country})'
#     else:
#         query = base_query
    
#     params = {
#         'q': query,
#         'language': 'en',
#         'sortBy': 'relevancy',
#         'from': start_date.strftime('%Y-%m-%d'),
#         'to': end_date.strftime('%Y-%m-%d'),
#         'apiKey': NEWS_API_KEY
#     }
    
#     try:
#         response = requests.get(NEWS_API_URL, params=params)
#         response.raise_for_status()
#         return response.json()['articles']
#     except Exception as e:
#         print(f"Error fetching news: {e}")
#         return []

# def analyze_news(articles, country=None):
#     """
#     Analyze news articles and provide a concise summary with sentiment analysis
#     """
#     food_crisis_count = 0
#     climate_crisis_count = 0
#     total_articles = len(articles)
    
#     if total_articles == 0:
#         if country:
#             result = f"No news articles found for {country} in the analyzed period."
#         else:
#             result = "No news articles found to analyze."
#         save_analysis_to_file(result, country)
#         return result
    
#     # Define comprehensive keyword categories with more specific terms
#     food_crisis_categories = {
#         'emergency': [
#             'famine', 'starvation', 'severe food shortage',
#             'acute food insecurity', 'food emergency'
#         ],
#         'global_threat': [
#             'global food crisis', 'world food crisis',
#             'food system collapse', 'food chain disruption'
#         ],
#         'humanitarian': [
#             'humanitarian food crisis', 'emergency food aid',
#             'food rationing', 'food vouchers'
#         ],
#         'production_crisis': [
#             'crop failure', 'agricultural collapse',
#             'harvest failure', 'food supply chain breakdown'
#         ],
#         'health_impact': [
#             'hunger-related deaths', 'severe malnutrition',
#             'acute undernourishment'
#         ]
#     }
    
#     climate_crisis_categories = {
#         'environmental': [
#             'climate crisis', 'climate emergency',
#             'climate breakdown', 'climate catastrophe'
#         ],
#         'natural_disasters': [
#             'extreme drought', 'severe flooding',
#             'devastating wildfire', 'catastrophic weather'
#         ],
#         'impact': [
#             'climate disaster', 'climate catastrophe',
#             'climate emergency', 'climate threat'
#         ],
#         'solutions': [
#             'climate action', 'climate solution',
#             'climate policy', 'climate goals'
#         ],
#         'scientific': [
#             'climate science', 'climate data',
#             'climate research', 'climate study'
#         ]
#     }
    
#     # Initialize category counters and sentiment trackers
#     food_category_counts = {category: 0 for category in food_crisis_categories}
#     climate_category_counts = {category: 0 for category in climate_crisis_categories}
#     food_sentiment_sum = 0
#     climate_sentiment_sum = 0
    
#     for article in articles:
#         title = article['title']
#         description = article['description']
#         content = article.get('content', '')
        
#         # Combine all text for analysis
#         text = f"{title} {description} {content}"
#         text = text.lower()
        
#         # Get sentiment score
#         blob = TextBlob(text)
#         sentiment_score = blob.sentiment.polarity
        
#         # Analyze food crisis categories
#         for category, keywords in food_crisis_categories.items():
#             if any(keyword in text for keyword in keywords):
#                 food_category_counts[category] += 1
#                 food_crisis_count += 1
#                 food_sentiment_sum += sentiment_score
#                 break
        
#         # Analyze climate crisis categories
#         for category, keywords in climate_crisis_categories.items():
#             if any(keyword in text for keyword in keywords):
#                 climate_category_counts[category] += 1
#                 climate_crisis_count += 1
#                 climate_sentiment_sum += sentiment_score
#                 break
    
#     # Calculate average sentiments
#     food_avg_sentiment = food_sentiment_sum / food_crisis_count if food_crisis_count > 0 else 0
#     climate_avg_sentiment = climate_sentiment_sum / climate_crisis_count if climate_crisis_count > 0 else 0
    
#     # Generate concise summary
#     if country:
#         summary = f"Analysis Results for {country}:\n"
#     else:
#         summary = f"Analysis Results:\n"
    
#     summary += f"• Total articles analyzed: {total_articles}\n"
#     summary += f"• Food crisis related articles: {food_crisis_count} ({food_crisis_count/total_articles*100:.1f}% of total)\n"
#     summary += f"• Climate crisis related articles: {climate_crisis_count} ({climate_crisis_count/total_articles*100:.1f}% of total)\n\n"
    
#     if food_crisis_count == 0 and climate_crisis_count == 0:
#         if country:
#             summary += f"No significant food or climate crisis issues are currently being reported for {country}."
#         else:
#             summary += "Neither food nor climate crisis issues are currently receiving significant media attention."
#     else:
#         # Determine severity based on category distribution
#         food_emergency_ratio = (food_category_counts['emergency'] + food_category_counts['global_threat']) / food_crisis_count if food_crisis_count > 0 else 0
#         climate_impact_ratio = (climate_category_counts['impact'] + climate_category_counts['natural_disasters']) / climate_crisis_count if climate_crisis_count > 0 else 0
        
#         # Generate summary based on coverage, severity, and sentiment
#         if food_crisis_count > 0:
#             food_severity = "serious" if food_emergency_ratio > 0.3 else "moderate"
#             food_sentiment = "negative" if food_avg_sentiment < -0.1 else "positive" if food_avg_sentiment > 0.1 else "neutral"
#             summary += f"Food crisis issues are receiving {food_severity} media attention with a generally {food_sentiment} tone"
        
#         if food_crisis_count > 0 and climate_crisis_count > 0:
#             summary += " and "
        
#         if climate_crisis_count > 0:
#             climate_severity = "serious" if climate_impact_ratio > 0.3 else "moderate"
#             climate_sentiment = "negative" if climate_avg_sentiment < -0.1 else "positive" if climate_avg_sentiment > 0.1 else "neutral"
#             summary += f"climate crisis issues are receiving {climate_severity} media attention with a generally {climate_sentiment} tone"
        
#         summary += "."
    
#     # Save analysis to file
#     save_analysis_to_file(summary, country)
    
#     return summary

# def main():
#     # Fetch news articles
#     articles = fetch_news()
    
#     if not articles:
#         print("No articles found or error occurred while fetching news.")
#         return
    
#     # Analyze articles and get summary
#     summary = analyze_news(articles)
    
#     # Print summary
#     print("\nNews Analysis Summary:")
#     print("-" * 50)
#     print(summary)
#     print("-" * 50)

# if __name__ == "__main__":
#     main() 
    

import requests
from textblob import TextBlob
import re
from datetime import datetime, timedelta
import os
import json

# News API configuration
NEWS_API_KEY = '9c0199db22344707914c11329b9bf06a'
NEWS_API_URL = 'https://newsapi.org/v2/everything'

# Region to country code mapping
REGION_TO_COUNTRY = {
    'sanliurfa': 'tr',      # Turkey
    'punjab': 'in',         # India
    'munich': 'de',         # Germany
    'iowa': 'us',           # United States
    'kano': 'ng',           # Nigeria
    'zacatecas': 'mx',      # Mexico
    'gauteng': 'za',        # South Africa
    'addis_ababa': 'et',    # Ethiopia
    'yunnan': 'cn',         # China
    'gujarat': 'in',        # India
    'cordoba': 'ar',        # Argentina
    'mato_grosso': 'br',    # Brazil
    'nsw': 'au'             # Australia
}

def get_country_code(region):
    """
    Get the country code for a given region
    """
    return REGION_TO_COUNTRY.get(region.lower())

def save_analysis_to_file(analysis_result: dict, country: str = None) -> str:
    """
    Save the analysis_result dict as pretty-printed JSON to a .txt file.
    Returns the filepath.
    """
    # Proje kök dizini
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir  = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname     = f"news_analysis_{country}_{timestamp}.txt" if country else f"news_analysis_{timestamp}.txt"
    path      = os.path.join(results_dir, fname)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)

    print(f"Analysis saved to: {path}")
    return path

def fetch_news(region=None):
    """
    Fetch news articles from News API with specific date range and relevance
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Get country code from region
    country_code = get_country_code(region) if region else None
    
    # Base query for food and climate crisis
    base_query = '(food crisis OR climate crisis) AND (impact OR effect OR threat OR risk OR solution)'
    
    # Add country to query if specified
    if country_code:
        query = f'({base_query}) AND (country:{country_code})'
    else:
        query = base_query
    
    params = {
        'q': query,
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

def analyze_news(articles: list, country: str = None) -> dict:
    """
    Analyze and return a dict with structured results.
    """
    total_articles      = len(articles)
    food_crisis_count   = 0
    climate_crisis_count= 0

    # Define comprehensive keyword categories
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

    food_category_counts    = {k: 0 for k in food_crisis_categories}
    climate_category_counts = {k: 0 for k in climate_crisis_categories}
    food_sentiment_sum      = 0.0
    climate_sentiment_sum   = 0.0

    # Store article details for UI display
    analyzed_articles = []

    for art in articles:
        text = f"{art.get('title','')} {art.get('description','')} {art.get('content','')}".lower()
        polarity = TextBlob(text).sentiment.polarity

        # Store article details
        article_info = {
            'title': art.get('title', ''),
            'description': art.get('description', ''),
            'url': art.get('url', ''),
            'publishedAt': art.get('publishedAt', ''),
            'sentiment': round(polarity, 3),
            'categories': []
        }

        # Food categories
        for cat, kws in food_crisis_categories.items():
            if any(kw in text for kw in kws):
                food_category_counts[cat] += 1
                food_crisis_count += 1
                food_sentiment_sum += polarity
                article_info['categories'].append(f'food_{cat}')
                break

        # Climate categories
        for cat, kws in climate_crisis_categories.items():
            if any(kw in text for kw in kws):
                climate_category_counts[cat] += 1
                climate_crisis_count += 1
                climate_sentiment_sum += polarity
                article_info['categories'].append(f'climate_{cat}')
                break

        analyzed_articles.append(article_info)

    # Calculate averages and percentages
    food_avg_sent    = (food_sentiment_sum / food_crisis_count) if food_crisis_count else 0.0
    climate_avg_sent = (climate_sentiment_sum / climate_crisis_count) if climate_crisis_count else 0.0
    food_pct    = round(food_crisis_count/total_articles*100,1) if total_articles else 0
    climate_pct = round(climate_crisis_count/total_articles*100,1) if total_articles else 0

    # Calculate severity ratios
    fe_ratio = (food_category_counts['emergency'] + food_category_counts['global_threat']) / food_crisis_count if food_crisis_count else 0
    ci_ratio = (climate_category_counts['impact'] + climate_category_counts['natural_disasters']) / climate_crisis_count if climate_crisis_count else 0

    def severity(r): return "serious" if r>0.3 else "moderate"
    def tone(s):     return "negative" if s<-0.1 else "positive" if s>0.1 else "neutral"

    result = {
        "country": country,
        "total_articles": total_articles,
        "food_crisis": {
            "count": food_crisis_count,
            "percent": food_pct,
            "category_counts": food_category_counts,
            "average_sentiment": round(food_avg_sent,3),
            "severity": severity(fe_ratio),
            "tone": tone(food_avg_sent)
        },
        "climate_crisis": {
            "count": climate_crisis_count,
            "percent": climate_pct,
            "category_counts": climate_category_counts,
            "average_sentiment": round(climate_avg_sent,3),
            "severity": severity(ci_ratio),
            "tone": tone(climate_avg_sent)
        },
        "articles": analyzed_articles
    }

    # Save analysis to file
    save_analysis_to_file(result, country)
    return result

def main():
    articles = fetch_news()      # veya fetch_news("Turkey") gibi
    if not articles:
        print("No articles found.")
        return

    analysis = analyze_news(articles)
    # Konsola da JSON olarak basabiliriz:
    print(json.dumps(analysis, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
