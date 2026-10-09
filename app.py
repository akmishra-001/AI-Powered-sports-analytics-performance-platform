import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import cv2
import tempfile
import os

from utils.cv_engine import SportsCVEngine

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Sports Analytics Platform",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ AI-Powered Sports Analytics Platform")
st.markdown("##### *Real-time Player Tracking, Outcome Prediction & Performance Forecasting Dashboard*")
st.divider()

# Session State Variables Setup
if "stamina_val" not in st.session_state:
    st.session_state.stamina_val = 80.0
if "pass_acc_val" not in st.session_state:
    st.session_state.pass_acc_val = 85.0
if "dist_val" not in st.session_state:
    st.session_state.dist_val = 10.5
if "speed_val" not in st.session_state:
    st.session_state.speed_val = 28.5
if "key_actions_val" not in st.session_state:
    st.session_state.key_actions_val = 5

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "cached_final_frame" not in st.session_state:
    st.session_state.cached_final_frame = None
if "cached_player_metrics" not in st.session_state:
    st.session_state.cached_player_metrics = {}

# ==========================================
# LOAD TRAINED MODELS & SCALERS (Optimized for Cloud RAM)
# ==========================================
@st.cache_resource
def load_all_assets():
    perf_model = keras.models.load_model("models/performance_model.keras", compile=False)
    perf_scaler = joblib.load("models/scaler_performance.pkl")
    
    match_model = keras.models.load_model("models/match_outcome_model.keras", compile=False)
    match_scaler = joblib.load("models/scaler_match.pkl")
    
    lstm_model = keras.models.load_model("models/forecasting_lstm.keras", compile=False)
    lstm_scaler = joblib.load("models/scaler_lstm.pkl")
    
    return perf_model, perf_scaler, match_model, match_scaler, lstm_model, lstm_scaler

try:
    perf_model, perf_scaler, match_model, match_scaler, lstm_model, lstm_scaler = load_all_assets()
    st.sidebar.success("✅ Sabhi Trained Models successfully load ho gaye hain!")
except Exception as e:
    st.sidebar.error("❌ Models load nahi ho paye! Pehle `python train_models.py` run karein.")

# NOTE: cv_engine yahan global initialize nahi hota taaki app start par RAM crash (Segmentation Fault) na ho.

# Sidebar Navigation Guide
st.sidebar.header("📌 Project Navigation Guide")
st.sidebar.info(
    "Neeche diye gaye Tabs mein se kisi bhi module ko choose karke testing kar sakte hain:\n"
    "- **Tab 1:** Live CV Analytics & Non-Blocking Player Selection\n"
    "- **Tab 2:** Player Performance Rating Evaluator (ANN)\n"
    "- **Tab 3:** Match Outcome Prediction (ANN)\n"
    "- **Tab 4:** Future Performance Forecasting (LSTM)\n"
    "- **Tab 5:** Dataset Explorer"
)

# MULTI-TAB NAVIGATION
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📹 Submodule 1 & 2: CV Player Tracking & Action",
    "📊 Submodule 3: Performance Prediction (ANN)",
    "🏆 Submodule 4: Match Outcome Prediction (ANN)",
    "📈 Submodule 5: Future Forecasting (LSTM)",
    "📁 Dataset Explorer"
])

# ------------------------------------------
# TAB 1: COMPUTER VISION ENGINE (NON-BLOCKING SELECTION & LAZY LOADED)
# ------------------------------------------
with tab1:
    st.subheader("📹 Computer Vision: Live Player Detection & Analytics")
    st.caption("👉 Live detection chalti dikhegi. Stop button se video freeze hoke player selection dropdown smooth chalega.")

    # Lazy initialization of CV Engine inside Tab 1 to prevent startup RAM crash
    @st.cache_resource
    def get_cv_engine():
        return SportsCVEngine()
    
    cv_engine = get_cv_engine()

    uploaded_video = st.file_uploader("Upload a Match Video Clip (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        col_btn1, col_btn2 = st.columns([1, 4])
        start_process = col_btn1.button("▶️ Start Video Detection")
        
        if start_process:
            st.session_state.video_processed = False
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            metric_players = st.metric("Active Players Detected", "0")
            
            stop_btn = st.button("🛑 Stop Video Processing")
            
            player_stats_db = {}
            frame_counter = 0
            last_frame = None

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_counter += 1
                annotated_frame, player_count = cv_engine.process_frame_and_track(frame, frame_counter, player_stats_db)
                
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                last_frame = annotated_frame_rgb
                
                st_frame.image(annotated_frame_rgb, channels="RGB")
                metric_players.metric("Active Players Detected", str(player_count))

            cap.release()

            # Save processed data to session_state to prevent app freeze
            final_metrics = cv_engine.calculate_final_player_metrics(player_stats_db)
            st.session_state.cached_player_metrics = final_metrics
            st.session_state.cached_final_frame = last_frame
            st.session_state.video_processed = True

        # Render Cached Analytics (Non-blocking Dropdown interaction)
        if st.session_state.video_processed and st.session_state.cached_player_metrics:
            st.divider()
            st.success("✅ Video Stopped & Player Metrics Saved Successfully!")
            
            if st.session_state.cached_final_frame is not None:
                st.image(st.session_state.cached_final_frame, channels="RGB", caption="Final Captured Video Frame", use_container_width=True)
            
            st.subheader("🎯 Per-Player Extracted Analytics")
            
            metrics_dict = st.session_state.cached_player_metrics
            selected_player = st.selectbox(
                "📌 Select Target Player to Analyze:", 
                list(metrics_dict.keys()),
                key="player_selection_dropdown"
            )
            
            selected_stats = metrics_dict[selected_player]
            
            st.write(f"### Extracted Stats for **{selected_player}**:")
            c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
            c_m1.metric("Stamina Score", f"{selected_stats['stamina']:.1f}")
            c_m2.metric("Pass Accuracy", f"{selected_stats['pass_accuracy']:.1f}%")
            c_m3.metric("Distance Covered", f"{selected_stats['distance']} km")
            c_m4.metric("Max Speed", f"{selected_stats['speed']} km/h")
            c_m5.metric("Key Actions", f"{selected_stats['key_actions']}")
            
            if st.button(f"⚡ Send {selected_player}'s Stats to Submodule 3", key="sync_btn"):
                st.session_state.stamina_val = selected_stats['stamina']
                st.session_state.pass_acc_val = selected_stats['pass_accuracy']
                st.session_state.dist_val = selected_stats['distance']
                st.session_state.speed_val = selected_stats['speed']
                st.session_state.key_actions_val = selected_stats['key_actions']
                
                st.toast(f"✅ {selected_player}'s stats sent to Submodule 3! Switch to Tab 2 to check.", icon="🚀")

    else:
        st.info("ℹ️ Kripya testing ke liye koi video upload karein.")

# ------------------------------------------
# TAB 2: PLAYER PERFORMANCE RATING (ANN)
# ------------------------------------------
with tab2:
    st.subheader("📊 Player Performance Score Evaluator (ANN Regression)")
    st.caption("👉 Selected Player ke extracted stats yahan auto-fill hain. Direct prediction check karein.")

    c1, c2, c3 = st.columns(3)
    stamina = c1.slider("Stamina Score", 50.0, 100.0, float(st.session_state.stamina_val))
    pass_acc = c2.slider("Pass Accuracy (%)", 40.0, 100.0, float(st.session_state.pass_acc_val))
    dist = c3.number_input("Distance Covered (km)", 1.0, 20.0, float(st.session_state.dist_val))
    
    c4, c5 = st.columns(2)
    speed = c4.number_input("Max Speed (km/h)", 10.0, 40.0, float(st.session_state.speed_val))
    key_actions = c5.slider("Key Actions Count (Goals/Passes/Tackles)", 0, 15, int(st.session_state.key_actions_val))

    if st.button("Predict Performance Rating"):
        input_data = np.array([[stamina, pass_acc, dist, speed, key_actions]])
        scaled_input = perf_scaler.transform(input_data)
        raw_pred = perf_model.predict(scaled_input, verbose=0)[0][0]
        final_rating = np.clip(raw_pred, 1.0, 10.0)
        
        st.success(f"🎯 Predicted Player Rating: **{final_rating:.2f} / 10.0**")

# ------------------------------------------
# TAB 3: MATCH OUTCOME PREDICTION (ANN)
# ------------------------------------------
with tab3:
    st.subheader("🏆 Team Match Outcome Predictor (ANN Classification)")
    st.caption("👉 Team performance stats ke basis par Win vs Loss/Draw probability check karein.")

    m1, m2, m3 = st.columns(3)
    possession = m1.slider("Ball Possession (%)", 20.0, 80.0, 55.0)
    shots = m2.number_input("Shots on Target", 0, 25, 8)
    pass_succ = m3.slider("Pass Success (%)", 30.0, 95.0, 78.0)
    
    m4, m5 = st.columns(2)
    fouls = m4.number_input("Fouls Committed", 0, 30, 8)
    defense = m5.slider("Defense Rating", 5.0, 10.0, 7.5)

    if st.button("Predict Match Result"):
        match_input = np.array([[possession, shots, pass_succ, fouls, defense]])
        scaled_m = match_scaler.transform(match_input)
        prob = match_model.predict(scaled_m, verbose=0)[0][0]
        
        if prob >= 0.5:
            st.balloons()
            st.success(f"🎉 **VICTORY PREDICTED!** Win Probability: {prob * 100:.1f}%")
        else:
            st.warning(f"⚠️ **DRAW / DEFEAT LIKELY.** Win Probability: {prob * 100:.1f}%")

# ------------------------------------------
# TAB 4: FUTURE PERFORMANCE FORECASTING (LSTM)
# ------------------------------------------
with tab4:
    st.subheader("📈 Player Next Match Rating Forecast (LSTM)")
    st.caption("👉 Pichle 5 matches ke performance trend se aane wale match ki rating predict karein.")

    st.write("Enter Ratings of Previous 5 Matches:")
    l1, l2, l3, l4, l5 = st.columns(5)
    r1 = l1.number_input("Match 1", 1.0, 10.0, 6.5)
    r2 = l2.number_input("Match 2", 1.0, 10.0, 7.0)
    r3 = l3.number_input("Match 3", 1.0, 10.0, 6.8)
    r4 = l4.number_input("Match 4", 1.0, 10.0, 7.5)
    r5 = l5.number_input("Match 5", 1.0, 10.0, 8.0)

    if st.button("Forecast Next Match Performance"):
        ts_input = np.array([r1, r2, r3, r4, r5]).reshape(1, 5, 1)
        raw_next_pred = lstm_model.predict(ts_input, verbose=0)[0][0]
        final_next_pred = np.clip(raw_next_pred, 1.0, 10.0)
        
        st.info(f"🔮 Forecasted Rating for Match #6: **{final_next_pred:.2f} / 10.0**")

# ------------------------------------------
# TAB 5: DATASET EXPLORER
# ------------------------------------------
with tab5:
    st.subheader("📁 CSV Generated Datasets Quick View")
    st.caption("👉 System dwara generated synthetic datasets ki details check karein.")
    
    data_option = st.selectbox("Select Dataset to Inspect", [
        "data/player_performance_data.csv",
        "data/match_outcome_data.csv",
        "data/player_timeseries_data.csv"
    ])
    
    if os.path.exists(data_option):
        df_preview = pd.read_csv(data_option)
        st.dataframe(df_preview.head(15))