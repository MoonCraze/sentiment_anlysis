import json
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from model_loader.berta_models import load_finbert_sentiment_model

# === Step 1: Load your JSON file ===
json_path = r"C:\Users\visal Adikari\OneDrive\Desktop\Uni Sem 7\sentiment alaysis\data\preprocessed_data1.json"
with open(json_path, "r", encoding='utf-8') as f:
    data = json.load(f)

# === Step 2: Extract news text and timestamps ===
news_texts = []
timestamps = []

for entry in data:
    for embed in entry.get("embeds", []):
        if embed.get("type") == "rich":
            title = embed.get("title", "")
            description = embed.get("description", "")
            text = f"{title}. {description}".strip()
            if text:
                news_texts.append(text)
                timestamps.append(embed.get("timestamp", None))

# === Step 3: Load FinBERT ===
sentiment_model = load_finbert_sentiment_model()

# === Step 4: Get dominant sentiment for each news item ===
labels = []
for text in news_texts:
    result = sentiment_model(text[:512])[0]  # Only top sentiment
    labels.append(result['label'].upper())

# === Step 5: Create DataFrame and print output ===
df = pd.DataFrame({
    'text': news_texts,
    'dominant_sentiment': labels,
    'timestamp': pd.to_datetime(timestamps, errors='coerce')
})

# Print full DataFrame
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100):
    print(df)
