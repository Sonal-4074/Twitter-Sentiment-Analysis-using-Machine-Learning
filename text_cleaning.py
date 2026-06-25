import pandas as pd
import re

columns = ["target","id","date","flag","user","text"]

df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    names=columns
)

# Use only first 5 tweets for testing
sample = df["text"].head(5)

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text

for tweet in sample:
    print("\nOriginal:")
    print(tweet)

    print("\nCleaned:")
    print(clean_text(tweet))