# ============================================================================
# PROJECT: AI-Powered Sports Analytics Platform
# FILE: app.py
# DESCRIPTION: Complete Green Transparent Glassmorphic Theme Dashboard.
# ============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

from utils.cv_engine import SportsCVEngine

# -----------------------------------------------------------------------------
# Page setup and visual theme (Green Transparent Glassmorphic UI)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Sports Analytics Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --green-primary: #10b981;
        --green-bright: #34d399;
        --green-mint: #6ee7b7;
        --dark-emerald: #064e3b;
        --app-bg-start: #022c22;
        --app-bg-end: #021f18;
        --panel-bg: rgba(6, 78, 59, 0.45);
        --border-color: rgba(16, 185, 129, 0.35);
        --text-main: #ecfdf5;
        --text-muted: #a7f3d0;
    }
    
    /* Main App Background (Green Transparent Glass Gradient) */
    .stApp {
        background: radial-gradient(circle at 20% 20%, rgba(16, 185, 129, 0.18) 0%, #021f18 70%) !important;
        color: var(--text-main);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Bar */
    [data-testid="stHeader"] { 
        background: rgba(2, 44, 34, 0.60); 
        backdrop-filter: blur(12px); 
        -webkit-backdrop-filter: blur(12px);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(2, 31, 24, 0.80) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] * { 
        color: var(--text-main) !important; 
    }
    
    /* Hero Banner */
    .hero {
        padding: 32px 36px;
        border-radius: 18px;
        border: 1px solid var(--border-color);
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.55) 0%, rgba(2, 44, 34, 0.70) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(16, 185, 129, 0.12);
        margin: 8px 0 25px;
    }
    .hero h1 { 
        color: var(--green-bright); 
        font-size: clamp(1.65rem, 3vw, 2.45rem); 
        margin: 0 0 8px; 
        letter-spacing: .02em; 
        text-shadow: 0 0 15px rgba(52, 211, 153, 0.3);
    }
    .hero p { color: var(--text-muted); margin: 0; font-size: 1rem; }
    .eyebrow { color: var(--green-mint); font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 7px; }
    
    /* Transparent Green Panels */
    .panel {
        border: 1px solid var(--border-color);
        border-radius: 16px;
        background: var(--panel-bg);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin: 4px 0 16px;
    }
    
    /* Info & Warning Boxes */
    .info-box, .warning-box {
        padding: 13px 15px;
        border-radius: 10px;
        margin: 8px 0 14px;
        font-size: .92rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .info-box { 
        color: #d1fae5; 
        background: rgba(16, 185, 129, 0.12); 
        border: 1px solid rgba(16, 185, 129, 0.4); 
        border-left: 4px solid var(--green-bright); 
    }
    .warning-box { 
        color: #fef3c7; 
        background: rgba(245, 158, 11, 0.12); 
        border: 1px solid rgba(245, 158, 11, 0.4); 
        border-left: 4px solid #f59e0b; 
    }
    
    /* File Uploader UI */
    div[data-testid="stFileUploader"] {
        background: rgba(2, 44, 34, 0.5) !important;
        border: 2px dashed var(--green-primary) !important;
        border-radius: 14px !important;
        padding: 12px !important;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }
    
    /* Metric Display Box */
    .metric-box {
        background: rgba(2, 31, 24, 0.65);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--green-bright);
        border-radius: 12px;
        padding: 16px 18px;
        margin: 10px 0;
        color: var(--text-main);
        line-height: 1.8;
    }
    
    /* Primary CTA Buttons (Vibrant Green) */
    .stButton > button, .stDownloadButton > button {
        border: none;
        border-radius: 10px;
        min-height: 2.8rem;
        color: #022c22 !important;
        background: linear-gradient(135deg, var(--green-bright) 0%, var(--green-primary) 100%) !important;
        font-weight: 800;
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
        box-shadow: 0 4px 18px rgba(16, 185, 129, 0.4);
        width: 100%;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        color: #022c22 !important;
        filter: brightness(1.12);
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(52, 211, 153, 0.6);
    }
    
    /* Input Fields & Dropdowns */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: rgba(2, 44, 34, 0.75) !important;
        color: var(--text-main) !important;
        border-radius: 8px;
        border: 1px solid var(--border-color) !important;
        backdrop-filter: blur(10px);
    }
    div[data-baseweb="input"] input, div[data-baseweb="select"] span {
        color: var(--text-main) !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { 
        background: var(--green-bright); 
        box-shadow: 0 0 10px var(--green-bright);
    }
    
    /* Streamlit Metrics */
    [data-testid="stMetric"] {
        background: var(--panel-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] { color: var(--green-bright) !important; text-shadow: 0 0 10px rgba(52, 211, 153, 0.25); }
    
    h1, h2, h3, h4, p, label { color: var(--text-main); }
    .footer { text-align: center; color: var(--text-muted); font-size: .84rem; padding: 16px 0 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Model / computer-vision engine loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading computer-vision engine…")
def load_cv_engine():
    return SportsCVEngine()

@st.cache_resource(show_spinner="Loading available deep-learning models…")
def load_dl_models():
    models = {}
    assets = {
        "perf_model": ("models/performance_model.keras", keras.models.load_model),
        "perf_scaler": ("models/scaler_performance.pkl", joblib.load),
        "match_model": ("models/match_outcome_model.keras", keras.models.load_model),
        "match_scaler": ("models/scaler_match.pkl", joblib.load),
        "lstm_model": ("models/forecasting_lstm.keras", keras.models.load_model),
        "lstm_scaler": ("models/scaler_lstm.pkl", joblib.load),
    }
    for key, (asset_path, loader) in assets.items():
        try:
            if Path(asset_path).is_file():
                models[key] = loader(asset_path)
        except Exception as exc:
            models[f"{key}_error"] = str(exc)
    return models

try:
    cv_engine = load_cv_engine()
    cv_engine_error = None
except Exception as exc:
    cv_engine = None
    cv_engine_error = str(exc)

dl_models = load_dl_models()

# -----------------------------------------------------------------------------
# Per-session state
# -----------------------------------------------------------------------------
def init_state():
    defaults = {
        "final_tracked_metrics": {},
        "extracted_stamina": 82.0,
        "extracted_pass": 88.0,
        "extracted_dist": 11.2,
        "extracted_speed": 31.0,
        "extracted_actions": 25,
        "transfer_success": False,
        "video_bytes": None,
        "video_name": None,
        "run_live": False,
        "perf_result": None,
        "match_result": None,
        "forecast_result": None,
        "last_uploaded_name": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# -----------------------------------------------------------------------------
# Header and navigation
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Computer vision · Neural networks · Match intelligence</div>
      <h1>AI Sports Analytics Platform</h1>
      <p>Track players, explore match telemetry, evaluate performance, and forecast ratings in one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

modules = [
    "📹 CV Tracking & Telemetry",
    "📊 Performance Evaluator (ANN)",
    "🏟️ Match Outcome Predictor (ANN)",
    "📈 Rating Forecasting (LSTM)",
    "📂 Dataset Explorer",
]
st.sidebar.markdown("## 🎛️ Control room")
st.sidebar.caption("Choose an analytics module")
selected_module = st.sidebar.radio("Analytics module", modules, label_visibility="collapsed")
st.sidebar.markdown("---")
if cv_engine_error:
    st.sidebar.warning("CV engine unavailable")
else:
    st.sidebar.success("Platform ready")

# -----------------------------------------------------------------------------
# Module 1: Video tracking and telemetry
# -----------------------------------------------------------------------------
if selected_module == "📹 CV Tracking & Telemetry":
    st.subheader("🎥 Player detection and telemetry")
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Upload match footage")
        uploaded_video = st.file_uploader(
            "Choose an MP4, AVI, or MOV file",
            type=["mp4", "avi", "mov"],
            key="match_video_uploader",
            help="The file is processed on this app's server.",
        )
        
        if uploaded_video is not None:
            if st.session_state.last_uploaded_name != uploaded_video.name:
                st.session_state.video_bytes = uploaded_video.getvalue()
                st.session_state.video_name = uploaded_video.name
                st.session_state.last_uploaded_name = uploaded_video.name
                st.session_state.final_tracked_metrics = {}
                st.session_state.transfer_success = False
                st.session_state.run_live = False

        video_bytes = st.session_state.video_bytes
        video_name = st.session_state.video_name
        if video_bytes:
            st.video(video_bytes)
            st.caption(f"Selected file: {video_name}")
            action_left, action_right = st.columns(2)
            with action_left:
                start_detection = st.button("▶ Start analysis")
            with action_right:
                stop_detection = st.button("⏹ Stop analysis")
            
            if start_detection:
                st.session_state.run_live = True
            if stop_detection:
                st.session_state.run_live = False
        else:
            st.markdown('<div class="warning-box">Upload a match clip first. Supported formats: MP4, AVI, and MOV.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Detection stream")
        
        if video_bytes and st.session_state.run_live:
            if cv_engine is None:
                st.error("Tracking cannot run because the CV engine did not load.")
            else:
                cap = None
                frame_placeholder = st.empty()
                progress_bar = st.progress(0.0)
                status = st.empty()
                player_stats = {}
                processed_count = 0
                last_annotated = None
                frame_skip = 2
                try:
                    with tempfile.NamedTemporaryFile(suffix=Path(video_name or "clip.mp4").suffix or ".mp4", delete=False) as temp_video:
                        temp_video.write(video_bytes)
                        video_path = temp_video.name
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("OpenCV could not open this video.")
                    else:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                        frame_idx = 0
                        
                        while cap.isOpened() and st.session_state.run_live:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            frame_idx += 1
                            if frame_idx % frame_skip == 0 or last_annotated is None:
                                try:
                                    annotated, active_count = cv_engine.process_frame_and_track(frame, frame_idx, player_stats)
                                    last_annotated = annotated if annotated is not None else frame
                                    processed_count += 1
                                except Exception as exc:
                                    st.error(f"Frame processing error: {exc}")
                                    break
                            display_frame = last_annotated if last_annotated is not None else frame
                            frame_placeholder.image(display_frame, channels="BGR", use_container_width=True)
                            if total_frames:
                                progress_bar.progress(min(frame_idx / total_frames, 1.0))
                                status.caption(f"Processing frame {frame_idx:,} of {total_frames:,}")
                            else:
                                status.caption(f"Processing frame {frame_idx:,}")
                        
                        if cap is not None and cap.isOpened():
                            cap.release()
                        cap = None
                        
                        if player_stats:
                            final_metrics = cv_engine.calculate_final_player_metrics(player_stats)
                            st.session_state.final_tracked_metrics = final_metrics or {}
                            st.session_state.run_live = False
                            st.rerun()
                except Exception as exc:
                    st.error(f"Video analysis failed: {exc}")
                finally:
                    if cap is not None:
                        cap.release()
                    if "video_path" in locals() and os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                        except OSError:
                            pass
                    st.session_state.run_live = False
        else:
            if not st.session_state.final_tracked_metrics:
                st.markdown('<div class="info-box">Upload a clip, then select <b>Start analysis</b>. Player metrics will appear here when tracking finishes.</div>', unsafe_allow_html=True)

        detected_players = list(st.session_state.final_tracked_metrics.keys())
        if detected_players:
            st.success(f"Analysis complete! Detected players: {len(detected_players)}")
            st.markdown("---")
            st.markdown('<div class="info-box">✨ Select any detected player below to inspect metrics and transfer data to Submodule 3.</div>', unsafe_allow_html=True)
            
            selected_player = st.selectbox("Choose detected player from analytical stream", detected_players, key="tracked_player_select")
            p_info = st.session_state.final_tracked_metrics[selected_player]
            
            st.markdown(
                f"""<div class="metric-box">📊 <b>{selected_player} Verified Telemetry Metrics:</b><br>
                • Stamina Index: <b>{p_info.get('stamina', 'N/A')}</b> | Pass Accuracy: <b>{p_info.get('pass_accuracy', 'N/A')}%</b><br>
                • Distance Covered: <b>{p_info.get('distance', 'N/A')} km</b> | Max Velocity: <b>{p_info.get('speed', 'N/A')} km/h</b><br>
                • Key Actions: <b>{p_info.get('key_actions', 'N/A')}</b></div>""",
                unsafe_allow_html=True,
            )
            
            if st.button("🚀 Transfer Telemetry to Submodule 3", key="transfer_telemetry"):
                try:
                    st.session_state.extracted_stamina = float(p_info["stamina"])
                    st.session_state.extracted_pass = float(p_info["pass_accuracy"])
                    st.session_state.extracted_dist = float(p_info["distance"])
                    st.session_state.extracted_speed = float(p_info["speed"])
                    st.session_state.extracted_actions = int(p_info["key_actions"])
                    
                    st.session_state.transfer_success = True
                    st.success(f"✅ {selected_player} telemetry data successfully synced with Submodule 3! Ab Sidebar se Module 3 par jaakar result dekhein.")
                except (KeyError, TypeError, ValueError) as exc:
                    st.error(f"Cannot transfer incomplete telemetry: {exc}")
        else:
            if not st.session_state.run_live:
                st.markdown(
                    """<div class="warning-box">
                    ⚠️ <b>Player selection locked:</b> Pehle video upload karke live detection run karein aur stream ko stop/complete karein.
                    </div>""",
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Module 2: Performance ANN
# -----------------------------------------------------------------------------
elif selected_module == "📊 Performance Evaluator (ANN)":
    st.subheader("📊 Player performance evaluator")
    st.caption("Estimate a player rating from physical and match-action metrics.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    if st.session_state.transfer_success:
        st.markdown('<div class="info-box">🔄 Telemetry data successfully received from CV Tracking module!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">💡 Tip: Aap yahan sliders ke through manual parameters bhi adjust kar sakte hain.</div>', unsafe_allow_html=True)

    stamina_default = float(np.clip(st.session_state.extracted_stamina, 50.0, 100.0))
    pass_default = float(np.clip(st.session_state.extracted_pass, 60.0, 98.0))
    dist_default = float(np.clip(st.session_state.extracted_dist, 4.0, 20.0))
    speed_default = float(np.clip(st.session_state.extracted_speed, 18.0, 50.0))
    actions_default = int(np.clip(st.session_state.extracted_actions, 5, 50))
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        stamina = st.slider("Stamina Score", 50.0, 100.0, stamina_default, key="perf_stamina")
        pass_acc = st.slider("Pass Accuracy (%)", 60.0, 98.0, pass_default, key="perf_pass")
        distance = st.number_input("Distance Covered (km)", 4.0, 20.0, dist_default, step=0.1, key="perf_distance")
    with c2:
        speed = st.number_input("Max Speed (km/h)", 18.0, 50.0, speed_default, step=0.5, key="perf_speed")
        actions = st.slider("Key Actions Count", 5, 50, actions_default, key="perf_actions")

    if st.button("Evaluate Performance Rating", key="evaluate_performance"):
        try:
            if "perf_model" in dl_models and "perf_scaler" in dl_models:
                input_data = np.array([[stamina, pass_acc, distance, speed, actions]], dtype=float)
                scaled = dl_models["perf_scaler"].transform(input_data)
                pred = dl_models["perf_model"].predict(scaled, verbose=0)
                rating = float(np.asarray(pred).reshape(-1)[0])
                model_label = "Trained ANN"
            else:
                rating = (stamina * 0.25 + pass_acc * 0.35 + distance * 2.5 + speed * 0.8 + actions * 0.4) / 10.0
                model_label = "Demo heuristic"
            rating = round(float(np.clip(rating, 1.0, 10.0)), 2)
            st.session_state.perf_result = {"rating": rating, "model": model_label}
        except Exception as exc:
            st.error(f"Performance prediction failed: {exc}")
            
    if st.session_state.perf_result:
        result = st.session_state.perf_result
        st.metric("Predicted Player Performance Rating", f"{result['rating']} / 10")
        st.caption(f"Result source: {result['model']}")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Module 3: Match outcome ANN
# -----------------------------------------------------------------------------
elif selected_module == "🏟️ Match Outcome Predictor (ANN)":
    st.subheader("🏟️ Match outcome predictor")
    st.caption("Estimate the model's predicted outcome from team-level match statistics.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        possession = st.slider("Ball Possession (%)", 30.0, 70.0, 55.0, key="match_possession")
        shots = st.number_input("Shots on Target", 1, 15, 8, key="match_shots")
        pass_success = st.slider("Pass Success (%)", 55.0, 92.0, 78.0, key="match_pass")
    with c2:
        fouls = st.number_input("Fouls Committed", 2, 20, 8, key="match_fouls")
        defense = st.slider("Defense Rating", 50.0, 95.0, 75.0, key="match_defense")

    if st.button("Predict Match Result", key="predict_match"):
        try:
            features = np.array([[possession, shots, pass_success, fouls, defense]], dtype=float)
            if "match_model" in dl_models and "match_scaler" in dl_models:
                scaled = dl_models["match_scaler"].transform(features)
                raw = np.asarray(dl_models["match_model"].predict(scaled, verbose=0)).reshape(-1)
                win_probability = float(np.clip(raw[-1] if raw.size >= 2 else raw[0], 0.0, 1.0))
                source = "Trained ANN"
            else:
                score = (possession * 0.3) + (shots * 3.5) + (pass_success * 0.2) - (fouls * 1.2) + (defense * 0.25)
                win_probability = float(1 / (1 + np.exp(-np.clip((score - 55) / 15, -30, 30))))
                source = "Demo heuristic"
            outcome = "Win 🏆" if win_probability >= 0.5 else "Loss / Draw ⚠️"
            st.session_state.match_result = {"outcome": outcome, "probability": win_probability, "source": source}
        except Exception as exc:
            st.error(f"Match prediction failed: {exc}")
            
    if st.session_state.match_result:
        result = st.session_state.match_result
        st.metric("Predicted Outcome", result["outcome"], f"Confidence Level: {result['probability']:.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Module 4: LSTM rating forecast
# -----------------------------------------------------------------------------
elif selected_module == "📈 Rating Forecasting (LSTM)":
    st.subheader("📈 Next-match rating forecast")
    st.caption("Provide five historical ratings in chronological order, oldest to newest.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    cols = st.columns(5)
    defaults = [6.5, 7.0, 6.8, 7.5, 8.0]
    ratings = [cols[i].number_input(f"Match {i + 1}", 1.0, 10.0, defaults[i], step=0.1, key=f"lstm_rating_{i}") for i in range(5)]

    if st.button("Forecast Next Match Rating", key="forecast_rating"):
        try:
            sequence = np.asarray(ratings, dtype=float).reshape(-1, 1)
            if "lstm_model" in dl_models:
                scaler = dl_models.get("lstm_scaler")
                model = dl_models["lstm_model"]
                prepared = scaler.transform(sequence) if scaler is not None else sequence
                prepared = prepared.reshape(1, 5, 1)
                raw_prediction = np.asarray(model.predict(prepared, verbose=0)).reshape(-1)
                prediction = float(raw_prediction[0])
                if scaler is not None and hasattr(scaler, "inverse_transform"):
                    prediction = float(scaler.inverse_transform(np.array([[prediction]])).reshape(-1)[0])
                source = "Trained LSTM"
            else:
                prediction = float(np.mean(ratings) + 0.2 * (ratings[-1] - ratings[0]))
                source = "Demo trend estimate"
            prediction = round(float(np.clip(prediction, 1.0, 10.0)), 2)
            st.session_state.forecast_result = {"rating": prediction, "source": source}
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
            
    if st.session_state.forecast_result:
        result = st.session_state.forecast_result
        st.metric("Forecasted Rating for Upcoming Match", f"{result['rating']} / 10")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Module 5: Dataset explorer
# -----------------------------------------------------------------------------
elif selected_module == "📂 Dataset Explorer":
    st.subheader("📂 Training dataset explorer")
    st.caption("Preview the CSV files used by your training and validation workflows.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    dataset_paths = [
        "data/player_performance_data.csv",
        "data/match_outcome_data.csv",
        "data/player_timeseries_data.csv",
    ]
    selected_dataset = st.selectbox("Select a dataset", dataset_paths, key="dataset_choice")
    if os.path.isfile(selected_dataset):
        try:
            df = pd.read_csv(selected_dataset)
            st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not read dataset: {exc}")
    else:
        st.warning(f"Dataset not found: {selected_dataset}.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer">AI Sports Analytics Platform · Green Transparent Glass Edition · Streamlit, OpenCV and TensorFlow</div>',
    unsafe_allow_html=True,
)