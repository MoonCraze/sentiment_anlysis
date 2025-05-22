import re
import json
import os

# 1. Locate & load raw tweets
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(script_dir, '..', 'data', 'preprocessed_data1.json')
with open(raw_path, 'r') as f:
    tweets = json.load(f)

# 2. Regex to capture $COIN +$1.2M / –$300K
pattern = r'\$(\w+)\s*([+-])\$(\d+(?:\.\d+)?[KM]?)'

coin_data = {}
for tweet in tweets:
    for coin, sign, value_str in re.findall(pattern, tweet):
        # multiplier for K/M
        if value_str.endswith('K'):
            multiplier, numeric = 1_000, value_str[:-1]
        elif value_str.endswith('M'):
            multiplier, numeric = 1_000_000, value_str[:-1]
        else:
            multiplier, numeric = 1, value_str

        try:
            val = float(numeric) * multiplier
        except ValueError:
            continue

        net = val if sign == '+' else -val
        coin_data.setdefault(coin, []).append(net)

# 3. Aggregate per coin
aggregated = {c: sum(vals) for c, vals in coin_data.items()}

# 4. Write out
out = {
    "detailed_flows": coin_data,
    "aggregated_flows": aggregated
}
out_path = os.path.join(script_dir, '..', 'data', 'coin_flow_data.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=4)

print(f"Saved aggregated flows to {out_path}")
