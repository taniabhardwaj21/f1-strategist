import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("Loading dataset...")

df = pd.read_csv("data/ml_dataset.csv")

# Split by race (avoid leakage)
races = df["Race"].unique()

train_races, test_races = train_test_split(
    races,
    test_size=0.2,
    random_state=42
)

train_df = df[df["Race"].isin(train_races)]
test_df = df[df["Race"].isin(test_races)]

X_train = train_df.drop(
    columns=["PitNextLap", "Race"]
)

y_train = train_df["PitNextLap"]

X_test = test_df.drop(
    columns=["PitNextLap", "Race"]
)

y_test = test_df["PitNextLap"]

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete.")

preds = model.predict(X_test)

print("\nResults:\n")

print(
    classification_report(
        y_test,
        preds
    )
)