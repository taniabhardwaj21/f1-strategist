import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/ml_dataset.csv")

races = df["Race"].unique()

train_races, test_races = train_test_split(
    races,
    test_size=0.2,
    random_state=42
)

train_df = df[df["Race"].isin(train_races)]

X_train = train_df.drop(
    columns=["PitNextLap", "Race"]
)

y_train = train_df["PitNextLap"]

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance)