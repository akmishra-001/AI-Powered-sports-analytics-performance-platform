import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure models and utils folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("utils", exist_ok=True)

print("🚀 Starting Phase 2: Deep Learning Models Training...\n")

# ==========================================
# 1. SUBMODULE 3: Player Performance Prediction (ANN Regression)
# ==========================================
print("--- [1/3] Training Submodule 3: Player Performance Prediction (ANN) ---")
df_perf = pd.read_csv("data/player_performance_data.csv")

X_perf = df_perf[["stamina_score", "pass_accuracy", "distance_covered_km", "speed_max_kmh", "key_actions_count"]].values
y_perf = df_perf["performance_rating"].values

# Standardize Features
scaler_perf = StandardScaler()
X_perf_scaled = scaler_perf.fit_transform(X_perf)
joblib.dump(scaler_perf, "models/scaler_performance.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_perf_scaled, y_perf, test_size=0.2, random_state=42)

# ANN Architecture
model_perf = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1) # Linear output for continuous rating
])

model_perf.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_perf.fit(X_train, y_train, epochs=40, batch_size=16, validation_split=0.1, verbose=0)

loss, mae = model_perf.evaluate(X_test, y_test, verbose=0)
print(f"✅ Submodule 3 ANN Model Trained | Test MAE: {mae:.2f}")
model_perf.save("models/performance_model.keras")
print("💾 Saved: models/performance_model.keras\n")


# ==========================================
# 2. SUBMODULE 4: Match Outcome Prediction (ANN Classification)
# ==========================================
print("--- [2/3] Training Submodule 4: Match Outcome Prediction (ANN) ---")
df_match = pd.read_csv("data/match_outcome_data.csv")

X_match = df_match[["possession_pct", "shots_on_target", "pass_success_pct", "fouls_committed", "defense_rating"]].values
y_match = df_match["match_outcome"].values

# Standardize Features
scaler_match = StandardScaler()
X_match_scaled = scaler_match.fit_transform(X_match)
joblib.dump(scaler_match, "models/scaler_match.pkl")

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_match_scaled, y_match, test_size=0.2, random_state=42)

# ANN Classification Architecture
model_match = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_m.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary output (Win: 1, Loss/Draw: 0)
])

model_match.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_match.fit(X_train_m, y_train_m, epochs=40, batch_size=16, validation_split=0.1, verbose=0)

m_loss, accuracy = model_match.evaluate(X_test_m, y_test_m, verbose=0)
print(f"✅ Submodule 4 ANN Model Trained | Test Accuracy: {accuracy*100:.2f}%")
model_match.save("models/match_outcome_model.keras")
print("💾 Saved: models/match_outcome_model.keras\n")


# ==========================================
# 3. SUBMODULE 5: Player Performance Forecasting (LSTM Sequential)
# ==========================================
print("--- [3/3] Training Submodule 5: Performance Forecasting (LSTM) ---")
df_ts = pd.read_csv("data/player_timeseries_data.csv")

# Create Sequence Data (Lookback = 5 matches to predict next match)
lookback = 5
X_lstm, y_lstm = [], []

for player_id, group in df_ts.groupby("player_id"):
    ratings = group["match_performance"].values
    if len(ratings) > lookback:
        for i in range(len(ratings) - lookback):
            X_lstm.append(ratings[i : i + lookback])
            y_lstm.append(ratings[i + lookback])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# Reshape for LSTM [samples, time_steps, features]
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

# Scale LSTM Target/Features
scaler_lstm = StandardScaler()
X_lstm_flat = X_lstm.reshape(-1, 1)
scaler_lstm.fit(X_lstm_flat)
joblib.dump(scaler_lstm, "models/scaler_lstm.pkl")

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# LSTM Architecture
model_lstm = keras.Sequential([
    layers.LSTM(64, activation='relu', input_shape=(lookback, 1), return_sequences=False),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_lstm.fit(X_train_l, y_train_l, epochs=50, batch_size=16, validation_split=0.1, verbose=0)

l_loss, l_mae = model_lstm.evaluate(X_test_l, y_test_l, verbose=0)
print(f"✅ Submodule 5 LSTM Model Trained | Test MAE: {l_mae:.2f}")
model_lstm.save("models/forecasting_lstm.keras")
print("💾 Saved: models/forecasting_lstm.keras\n")

print("🎉 Phase 2 Complete: All Deep Learning Models Trained & Saved successfully!")