import json
import os

def extract_text(tweet):
    text = tweet.get("content", "")
    for embed in tweet.get("embeds", []):
        text += " " + embed.get("title", "")
        text += " " + embed.get("description", "")
        text += " " + embed.get("timestamp", "")
        # Safely extract author name if available
        author = embed.get("author", {}).get("name", "")
        text += " " + author
    return text.strip()

def preprocess_data(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [extract_text(tweet) for tweet in data]
    with open(output_path, "w") as f:
        json.dump(texts, f, indent=2)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_path = os.path.join("..", "data", "x_data.json")
    output_path = os.path.join("..", "data", "preprocessed_data1.json")
    preprocess_data(input_path, output_path)
