import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

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
model_name = "cardiffnlp/twitter-roberta-base-sentiment"  # This is better for tweet sentiment
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

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
