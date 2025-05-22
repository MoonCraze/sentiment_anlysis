import re
import json
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 1. Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# 2. Load tweets
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_path  = os.path.join(script_dir, '..', 'data', 'preprocessed_data1.json')
with open(raw_path, 'r', encoding='utf-8') as f:
    tweets = json.load(f)

# 3. Prepare tokenizer & sentiment analyzer
pattern = r'\$(\w+)'  # just capture coin symbols
analyzer = SentimentIntensityAnalyzer()

# 4. Accumulate sentiment scores per coin
sentiment_data = {}     # coin -> [scores]
for tweet in tweets:
    # find all coins mentioned in this tweet
    coins = re.findall(pattern, tweet)
    if not coins:
        continue

    # compute a compound sentiment score for the tweet
    score = analyzer.polarity_scores(tweet)['compound']

    # assign this score to every coin found
    for coin in coins:
        sentiment_data.setdefault(coin, []).append(score)

# 5. Average scores per coin
averaged = {
    coin: sum(scores) / len(scores)
    for coin, scores in sentiment_data.items()
}

# 6. Write out sentiment_data.json
out_path = os.path.join(script_dir, '..', 'data', 'sentiment_data.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(averaged, f, indent=4)

print(f"Sentiment per coin saved to {out_path}")
