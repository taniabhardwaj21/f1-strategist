import pandas as pd

print("Loading data...")

df = pd.read_csv("data/season_2024_clean.csv")

# Sort laps properly
df = df.sort_values(
    ["Race", "Driver", "LapNumber"]
)

# Look at next lap's stint
df["NextStint"] = (
    df.groupby(["Race", "Driver"])["Stint"]
      .shift(-1)
)

# Target
df["PitNextLap"] = (
    df["NextStint"] > df["Stint"]
).astype(int)

df.drop(columns=["NextStint"], inplace=True)

print("\nLabel distribution:")
print(df["PitNextLap"].value_counts())

df.to_csv(
    "data/training_data.csv",
    index=False
)

print("\nSaved:")
print("data/training_data.csv")