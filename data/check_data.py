import pandas as pd

df = pd.read_csv("data/season_2024_clean.csv")

print(df.head())
print()
print(df.columns.tolist())
print()
print(df.shape)