import pandas as pd

df = pd.read_csv("data/season_2024_fixed.csv")

# Remove rows without lap times
df = df.dropna(subset=["LapTime"])

# Remove rows without position
df = df.dropna(subset=["Position"])

print("Remaining rows:", len(df))

df.to_csv(
    "data/season_2024_clean.csv",
    index=False
)