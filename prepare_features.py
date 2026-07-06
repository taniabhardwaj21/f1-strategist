import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("Loading data...")

df = pd.read_csv("data/training_data.csv")

# Convert LapTime to seconds
df["LapTime"] = pd.to_timedelta(
    df["LapTime"]
).dt.total_seconds()

features = [
    "Race",
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "TrackStatus",
    "Compound",
    "LapTime",
    "PitNextLap"
]

ml_df = df[features].copy()

# Encode Compound
compound_encoder = LabelEncoder()

ml_df["Compound"] = compound_encoder.fit_transform(
    ml_df["Compound"].astype(str)
)

# Encode TrackStatus
track_encoder = LabelEncoder()

ml_df["TrackStatus"] = track_encoder.fit_transform(
    ml_df["TrackStatus"].astype(str)
)

ml_df.to_csv(
    "data/ml_dataset.csv",
    index=False
)

print("\nSaved ml_dataset.csv")
print("\nDataset shape:")
print(ml_df.shape)

print("\nFirst 5 rows:")
print(ml_df.head())