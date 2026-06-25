import pandas as pd

columns = [
    "target",
    "id",
    "date",
    "flag",
    "user",
    "text"
]

df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    names=columns
)

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nSentiment Counts:")
print(df["target"].value_counts())