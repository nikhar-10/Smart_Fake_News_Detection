import pandas as pd

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real], ignore_index=True)
data = data.dropna(subset=["text"])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data.to_csv("data/news.csv", index=False)

print("Dataset prepared successfully!")
print("Total articles:", len(data))
print(data["label"].value_counts())