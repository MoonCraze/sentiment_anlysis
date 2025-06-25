import json
import os
import re
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

from model_loader.berta_models import load_deberta_ner_model

# Load NER-enabled DeBERTa model
ner_pipeline = load_deberta_ner_model()

# Load tweet JSON
tweets_file = r"C:\Users\visal Adikari\OneDrive\Desktop\Uni Sem 7\sentiment alaysis\data\preprocessed_data1.json"
with open(tweets_file, 'r', encoding='utf-8') as f:
    tweets = json.load(f)

# Extract text grouped by coin
coin_texts = {}

for tweet in tweets:
    if 'embeds' in tweet and tweet['embeds']:
        for embed in tweet['embeds']:
            text_parts = []
            if 'title' in embed and embed['title']:
                text_parts.append(embed['title'])
            if 'description' in embed and embed['description']:
                text_parts.append(embed['description'])
            combined_text = " ".join(text_parts)
            coins_found = re.findall(r'\$(\w+)', combined_text)
            for coin in coins_found:
                coin_texts[coin] = coin_texts.get(coin, "") + " " + combined_text

# Extract and count keywords using DeBERTa NER
coin_keywords_count = {}

for coin, text in coin_texts.items():
    ner_results = ner_pipeline(text)
    # Extract keyword text from NER
    keywords = [entity['word'] for entity in ner_results if entity['entity_group'] in ['ORG', 'MISC', 'PER', 'PRODUCT']]
    keyword_freq = dict(Counter(keywords))
    # Keep only keywords that appear more than 5 times
    filtered_keywords = {kw: count for kw, count in keyword_freq.items() if count > 5}
    coin_keywords_count[coin] = filtered_keywords

# Save results
output_data = {
    "coin_keywords_from_deberta": coin_keywords_count
}

script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, '..', 'data', 'coin_keywords_from_deberta.json')

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4)

print("Keyword frequencies from DeBERTa saved to", output_file)
