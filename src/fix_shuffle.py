import re

with open('fine_tune.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the dataset selection to shuffle
old_code = """        # Optional: Limit samples for faster training
        if max_samples:
            dataset["train"] = dataset["train"].select(range(max_samples))
            dataset["test"] = dataset["test"].select(range(max_samples // 5))"""

new_code = """        # Optional: Limit samples for faster training
        # CRITICAL: Shuffle before selecting to avoid class imbalance (IMDb is sorted by label!)
        if max_samples:
            dataset["train"] = dataset["train"].shuffle(seed=42).select(range(max_samples))
            dataset["test"] = dataset["test"].shuffle(seed=42).select(range(max_samples // 5))"""

content = content.replace(old_code, new_code)

with open('fine_tune.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed dataset shuffling to prevent class imbalance!")
