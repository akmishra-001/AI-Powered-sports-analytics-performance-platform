import os
import numpy as np
import pandas as pd

# Data directory create karne ke liye
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

np.random.seed(42)

print("⏳ Generating Datasets for Sports Analytics Platform...")

# ==========================================
# 1. SUBMODULE 3 DATASET: Player Performance Prediction (ANN)
# ==========================================
num_players = 1000
player_data = {
    "player_id": [f"P_{1000+i}" for i in range(num_players)],
    "stamina_score": np.random.uniform(50, 100, num_players),
    "pass_accuracy": np.random.uniform(60, 98, num_players),
    "distance_covered_km": np.random.uniform(4.0, 13.0, num_players),
    "speed_max_kmh": np.random.uniform(18.0, 35.0, num_players),
    "key_actions_count": np.random.randint(5, 50, num_players)
}
df_player = pd.DataFrame(player_data)

# Target Performance Rating (Formula + Small Noise)
df_player["performance_rating"] = (
    (df_player["stamina_score"] * 0.25) +
    (df_player["pass_accuracy"] * 0.35) +
    (df_player["distance_covered_km"] * 2.5) +
    (df_player["speed_max_kmh"] * 0.8) +
    (df_player["key_actions_count"] * 0.4)
) / 10.0
df_player["performance_rating"] = np.clip(df_player["performance_rating"], 1.0, 10.0).round(2)
df_player.to_csv("data/player_performance_data.csv", index=False)
print("✅ Saved: data/player_performance_data.csv (Submodule 3)")

# ==========================================
# 2. SUBMODULE 4 DATASET: Match Outcome Prediction (ANN)
# ==========================================
num_matches = 800
match_data = {
    "possession_pct": np.random.uniform(30, 70, num_matches),
    "shots_on_target": np.random.randint(1, 15, num_matches),
    "pass_success_pct": np.random.uniform(55, 92, num_matches),
    "fouls_committed": np.random.randint(2, 20, num_matches),
    "defense_rating": np.random.uniform(50, 95, num_matches)
}
df_match = pd.DataFrame(match_data)

# Match Outcome Logic: 1 = Win, 0 = Loss/Draw
score = (
    (df_match["possession_pct"] * 0.3) +
    (df_match["shots_on_target"] * 3.5) +
    (df_match["pass_success_pct"] * 0.2) -
    (df_match["fouls_committed"] * 1.2) +
    (df_match["defense_rating"] * 0.25)
)
median_score = np.median(score)
df_match["match_outcome"] = (score > median_score).astype(int)
df_match.to_csv("data/match_outcome_data.csv", index=False)
print("✅ Saved: data/match_outcome_data.csv (Submodule 4)")

# ==========================================
# 3. SUBMODULE 5 DATASET: Player Performance Forecasting (LSTM)
# ==========================================
# Time-series sequence data (50 players over 20 consecutive matches)
num_players_ts = 50
matches_count = 20
ts_rows = []

for p_id in range(num_players_ts):
    base_rating = np.random.uniform(5.5, 8.5)
    trend = np.random.uniform(-0.05, 0.08)
    for m in range(1, matches_count + 1):
        noise = np.random.normal(0, 0.3)
        rating = np.clip(base_rating + (m * trend) + noise, 3.0, 10.0)
        ts_rows.append({
            "player_id": f"P_{2000+p_id}",
            "match_num": m,
            "match_performance": round(rating, 2)
        })

df_ts = pd.DataFrame(ts_rows)
df_ts.to_csv("data/player_timeseries_data.csv", index=False)
print("✅ Saved: data/player_timeseries_data.csv (Submodule 5)")

print("\n🚀 Phase 1 Datasets successfully generated!")