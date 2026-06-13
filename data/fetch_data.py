import fastf1

fastf1.Cache.enable_cache("./cache")

session = fastf1.get_session(2024, "Monaco", "R")
session.load()

laps = session.laps

laps.to_csv("data/monaco_2024_race.csv", index=False)

print("Saved:", len(laps), "laps")