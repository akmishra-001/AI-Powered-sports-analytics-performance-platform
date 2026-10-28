# ⚽ AI-Powered Sports Analytics Platform

An enterprise-grade, full-stack AI sports analytics platform built with **Python**, **Streamlit**, **OpenCV**, and **TensorFlow**. 
This system delivers real-time player detection, computer vision telemetry tracking, deep learning performance evaluation, match outcome prediction, and time-series rating forecasting in a single workspace.

---

## 🚀 Key Features

* **📹 CV Tracking & Telemetry:** Real-time player detection, tracking, and metric calculation (Stamina Index, Pass Accuracy, Distance Covered, Max Velocity, and Key Actions) using OpenCV.
* **📊 Performance Evaluator (ANN):** Estimates player performance ratings from physical and match-action metrics using trained Artificial Neural Networks (ANN).
* **🏟️ Match Outcome Predictor (ANN):** Predicts match win/loss probabilities based on team-level statistics (Ball Possession, Shots on Target, Pass Success, Fouls, and Defense Rating).
* **📈 Rating Forecasting (LSTM):** Forecasts upcoming match ratings from historical player time-series data using Long Short-Term Memory (LSTM) networks.
* **📂 Dataset Explorer:** Interactive exploration and preview of training and validation CSV datasets.
* **💎 Custom Glassmorphic UI:** Modern, responsive green-transparent glassmorphic dashboard styled with custom CSS.

---

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit, Custom CSS (Glassmorphic Theme)
* **Computer Vision:** OpenCV, NumPy, Pandas
* **Deep Learning & Machine Learning:** TensorFlow, Keras, Scikit-learn, Joblib
* **Development Tools:** Git, GitHub, VS Code

---

## 📂 Project Structure

```text
ai-sports-analytics-platform/
│
├── app.py                  # Main Streamlit application dashboard
├── utils/
│   └── cv_engine.py        # OpenCV computer vision tracking engine
├── models/
│   ├── performance_model.keras
│   ├── match_outcome_model.keras
│   ├── forecasting_lstm.keras
│   └── (scalers and pkl files)
├── data/
│   ├── player_performance_data.csv
│   ├── match_outcome_data.csv
│   └── player_timeseries_data.csv
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


⚙️ Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/your-username/ai-sports-analytics-platform.git](https://github.com/your-username/ai-sports-analytics-platform.git)
cd ai-sports-analytics-platform

Create and activate a virtual environment:

Windows (PowerShell):

PowerShell
python -m venv venv
.\venv\Scripts\Activate
Windows (CMD):

DOS
python -m venv venv
venv\Scripts\activate.bat
Linux / macOS:

Bash
python3 -m venv venv
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Run the application:

Bash
streamlit run app.py

📊 Usage Guide
Open the local Streamlit link provided in your terminal (usually http://localhost:8501).

Use the Control Room sidebar to switch between different analytics modules:

Upload a match video clip (.mp4, .avi, .mov) in CV Tracking & Telemetry, run analysis, and sync tracked player metrics.

Evaluate player ratings or predict match outcomes using the ANN models.

Forecast future ratings using the LSTM module.

Explore underlying raw datasets in the Dataset Explorer.

👨‍💻 Author
Abhishek Mishra
