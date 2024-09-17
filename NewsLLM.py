from transformers import pipeline
import requests
from datetime import date, timedelta

# BART-based zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Categories for classification
categories = [
    "Geopolitics", 
    "Diplomacy", 
    "International Trade", 
    "Global Security", 
    "Foreign Policy", 
    "International Organizations", 
    "Transnational Issues",
    "War",
    "Human Rights",
    "Climate Change and Environment",
    "Humanitarian Aid",
    "Terrorism and Counterterrorism",
    "Nuclear Proliferation",
    "Cybersecurity",
    "Migration and Refugees",
    "Economic Sanctions",
    "International Law",
    "Cultural Diplomacy",
    "Public Policy",
    "United States Domestic Policy",
    "Other"
]
# Function to classify an article using BART
def classify_article_bart(article_content):
    result = classifier(article_content[:512], candidate_labels=categories, multi_label=False)
    return result['labels'][0]  # Return the highest-scoring label

# Set up NewsAPI parameters
API_KEY = "214c8ee3b0c74687909db7125f06aff2"
BASE_URL = 'https://newsapi.org/v2/everything'

# Get today's date
today = date.today()
# Calculate the date 30 days ago
thirty_days_ago = today - timedelta(days=30)

# Define search parameters for international relations news
params = {
    'q': 'international relations OR War Crimes OR UN OR United Nations OR human rights OR diplomacy OR foreign policy OR geopolitics OR global affairs OR world politics OR international cooperation OR global security OR international organizations OR transnational issues',
    'from': thirty_days_ago.isoformat(),  # Use the date from 30 days ago
    'to': today.isoformat(),              # End date (today)
    'sortBy': 'publishedAt',
    'language': 'en',
    'apiKey': API_KEY
}

# Make the API request to NewsAPI
response = requests.get(BASE_URL, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    # Extract the articles
    articles = data.get('articles', [])
    
    # Gather article content (or fallback to description/title)
    article_contents = []
    for article in articles:
        content = article.get('content') or article.get('description') or article.get('title')
        if content:
            article_contents.append(content)
else:
    print(f"Failed to retrieve articles: {response.status_code}")




# ... existing code ...

# Classify each article and print content with classification
for i, content in enumerate(article_contents):
    category = classify_article_bart(content)
    classifications.append(category)
    print(f"Article {i+1}:")
    print(f"Classification: {category}")
    print(f"Content: {content[:200]}...")  # Print first 200 characters of content
    print("-" * 50)  # Separator for readability

# Print a summary of how many articles fall into each category
for category in categories:
    count = classifications.count(category)
    print(f"{category}: {count} articles")

# Remove the line that prints article_contents[31]