import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from model_loader.berta_models import load_deberta_sentiment_model

# === Step 1: Load tweets from JSON ===
tweets_path = r"C:\Users\visal Adikari\OneDrive\Desktop\Uni Sem 7\sentiment alaysis\data\general_tweets.json"
with open(tweets_path, "r", encoding="utf-8") as f:
    tweets = json.load(f)

# === Step 2: Extract plain tweet texts ===
tweet_texts = []

for tweet in tweets:
    if isinstance(tweet, dict) and 'text' in tweet:
        tweet_texts.append(tweet['text'])
    elif isinstance(tweet, str):
        tweet_texts.append(tweet)

# === Step 3: Load DeBERTa v3 sentiment model ===
sentiment_pipeline = load_deberta_sentiment_model()

# === Step 4: Predict sentiment for each tweet ===
labels = []
for text in tweet_texts:
    result = sentiment_pipeline(text[:512])[0]  # Truncate to 512 tokens
    labels.append(result['label'].upper())

# === Step 5: Print Results ===
df = pd.DataFrame({
    'tweet': tweet_texts,
    'sentiment': labels
})

with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100):
    print(df)
