import json
import pandas as pd
from transformers import pipeline
from model_loader.berta_models import load_finbert_sentiment_model
from services.tweet_converter import run_preprocessing_news

# === Step 1: Get the path to the preprocessed news JSON ===
json_path = run_preprocessing_news()

# === Step 2: Load the list of news strings ===
with open(json_path, "r", encoding='utf-8') as f:
    news_texts = json.load(f)

# Optional: extract timestamps if embedded in the string (e.g., last 20-25 chars)
# But here we’ll just focus on sentiment analysis since timestamps aren't clearly separated

# === Step 3: Load FinBERT model ===
sentiment_model = load_finbert_sentiment_model()

# === Step 4: Run sentiment analysis and collect labels ===
labels = []
for text in news_texts:
    trimmed = text[:512]  # truncate to model input limit
    result = sentiment_model(trimmed)[0]
    labels.append(result["label"].upper())

# === Step 5: Create and print DataFrame ===
df = pd.DataFrame({
    "text": news_texts,
    "dominant_sentiment": labels,
    # If you extract timestamps from string, you can add them here
})

with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100):
    print(df)
