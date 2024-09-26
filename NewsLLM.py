from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import requests
from datetime import date, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import os

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
    "International Relations",
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
    "United States Domestic Policy"
]

# Function to classify an article using BART
def classify_article_bart(article_content):
    result = classifier(article_content[:512], candidate_labels=categories, multi_label=False)
    return result['labels'][0]  # Return the highest-scoring label

# Set up NewsAPI parameters
API_KEY = os.getenv('newsapikey')  # Ensure the environment variable is set
if not API_KEY:
    raise ValueError("API key not found. Make sure 'newsapikey' is set in the environment variables.")

BASE_URL = 'https://newsapi.org/v2/everything'

# Get today's date and calculate the date 30 days ago
today = date.today()
thirty_days_ago = today - timedelta(days=30)

# Define search parameters for international relations news
params = {
    'q': 'international relations OR War Crimes OR UN OR United Nations OR human rights OR diplomacy OR foreign policy OR geopolitics OR global affairs OR world politics OR international cooperation OR global security OR international organizations OR transnational issues',
    'from': thirty_days_ago.isoformat(),
    'to': today.isoformat(),
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
    raise RuntimeError(f"Failed to retrieve articles: {response.status_code}")

# Classify articles
classifications = []

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

# Initialize SentenceTransformer for encoding
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Encode article contents
article_embeddings = encoder.encode(article_contents)

# Initialize BART model for question answering
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def retrieve_relevant_articles(query, top_k=3):
    query_embedding = encoder.encode([query])
    similarities = cosine_similarity(query_embedding, article_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [article_contents[i] for i in top_indices]

def answer_question(question, context):
    inputs = tokenizer([context], [question], return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=300)
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def rag_qa(question):
    relevant_articles = retrieve_relevant_articles(question)
    context = " ".join(relevant_articles)
    answer = answer_question(question, context)
    return answer

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("Ask a question about the articles (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        answer = rag_qa(user_question)
        print(f"Answer: {answer}\n")
