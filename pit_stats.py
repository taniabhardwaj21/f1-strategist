import pandas as pd

df = pd.read_csv("data/training_data.csv")

pit_counts = (
    df.groupby("Race")["PitNextLap"]
      .sum()
      .sort_values(ascending=False)
)

print(pit_counts)

print("\nTotal pit labels:")
print(df["PitNextLap"].sum())