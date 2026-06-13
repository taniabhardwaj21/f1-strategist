import fastf1
import pandas as pd

fastf1.Cache.enable_cache("./cache")

YEAR = 2024

schedule = fastf1.get_event_schedule(YEAR)

all_laps = []

for _, event in schedule.iterrows():

    round_number = event["RoundNumber"]
    race_name = event["EventName"]

    if pd.isna(round_number):
        continue

    print(f"Loading Round {round_number}: {race_name}")

    try:
        session = fastf1.get_session(YEAR, round_number, "R")
        session.load()

        laps = session.laps.copy()

        laps["Race"] = race_name
        laps["Year"] = YEAR

        all_laps.append(laps)

        print(f"Collected {len(laps)} laps")

    except Exception as e:
        print(f"Failed {race_name}: {e}")

combined = pd.concat(all_laps, ignore_index=True)

combined.to_csv("data/season_2024_fixed.csv", index=False)

print(f"\nSaved {len(combined)} total laps")