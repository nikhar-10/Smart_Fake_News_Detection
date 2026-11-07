
import pandas as pd

data = pd.read_csv("data/True.csv")

keywords = ["NASA", "Moon", "lunar", "Chandrayaan"]

for keyword in keywords:
    results = data[
        data.astype(str).apply(
            lambda col: col.str.contains(
                keyword, case=False, na=False
            )
        ).any(axis=1)
    ]

    print(f"\n{'='*20} {keyword} {'='*20}")
    print("Articles found:", len(results))

    for _, row in results.head(5).iterrows():
        print("\nTitle:", row.get("title", ""))
        print("Text:", str(row.get("text", ""))[:300])
