import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
columns = ["target","id","date","flag","user","text"]

df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    names=columns
)

# Use first 50,000 rows
negative = df[df["target"] == 0].sample(25000, random_state=42)
positive = df[df["target"] == 4].sample(25000, random_state=42)

df = pd.concat([negative, positive])

df = df.sample(frac=1, random_state=42)


# Convert labels
df["target"] = df["target"].replace(4, 1)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# Features and labels
X = df["text"]
y = df["target"]

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(df["target"].value_counts())

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)