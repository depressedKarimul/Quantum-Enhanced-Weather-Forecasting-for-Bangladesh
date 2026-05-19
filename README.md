# QuantWeather-BD 🌦️⚛️

### Quantum Machine Learning-Based Weather Forecasting for Bangladesh

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-QML-674ea7.svg)](https://pennylane.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Webhook-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

QuantWeather-BD is a research-grade hybrid Quantum-Classical Machine Learning system for next-day weather forecasting in Dhaka, Bangladesh. It combines a Variational Quantum Circuit (VQC) implemented with PennyLane and PyTorch with classical neural layers, evaluates performance against conventional machine learning baselines, and deploys the resulting forecasts through a bilingual Facebook Messenger chatbot powered by Groq LLM.

---

## 📌 Overview

The project forecasts six next-day weather variables for Dhaka using 45 years of NASA POWER daily weather data. The core model is a hybrid QML architecture that encodes scaled weather features into a quantum circuit, learns entangled quantum representations, and maps quantum measurements back to weather variables through a classical neural network.

Alongside regression forecasting, QuantWeather-BD includes a Random Forest rain classifier with SMOTE oversampling, SHAP-based explainability for model transparency, and a FastAPI webhook for Facebook Messenger integration.

---

## 🧾 Dataset

| Property | Details |
|---|---|
| Source | NASA POWER Daily Weather Data |
| Location | Dhaka, Bangladesh |
| Date Range | 1981-2026 |
| Coverage | 45 years of daily observations |
| Input Features | T2M, T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, WS2M |
| Prediction Targets | All 6 features shifted by 1 day |

### Feature Set

| Feature | Description |
|---|---|
| T2M | Average temperature at 2 meters |
| T2M_MAX | Maximum temperature at 2 meters |
| T2M_MIN | Minimum temperature at 2 meters |
| RH2M | Relative humidity at 2 meters |
| PRECTOTCORR | Corrected precipitation |
| WS2M | Wind speed at 2 meters |

---

## 🧠 Model Architecture

### Hybrid QML Model

`HybridQMLModel` predicts six next-day weather variables from six scaled input features.

| Component | Configuration |
|---|---|
| Input | 6 MinMax-scaled weather features |
| Quantum Encoding | `AngleEmbedding` |
| Quantum Layer | `StronglyEntanglingLayers` |
| Qubits | 6 |
| Quantum Depth | 3 layers |
| Measurement | PauliZ expectation values |
| Classical Head | `Linear(6 -> 16)` + ReLU + `Linear(16 -> 6)` |
| Frameworks | PennyLane + PyTorch |
| Output | 6 next-day weather variables |

### Rain Classifier

| Component | Details |
|---|---|
| Model | Random Forest |
| Class Balancing | SMOTE oversampling |
| Classes | No Rain, Light, Moderate, Heavy |
| Output | Rain category and confidence percentage |

### SHAP Explainability

| Model | Explainer |
|---|---|
| Hybrid QML Model | `KernelExplainer` |
| Random Forest | `TreeExplainer` |
| Rain Classifier | SHAP feature importance |

SHAP is used to identify the top three features influencing each prediction and to translate model behavior into plain-language explanations for chatbot users.

### Baseline Models

QuantWeather-BD compares the QML model against:

- Linear Regression
- Random Forest Regressor
- MLP classical neural network
- LSTM neural network

---

## ⚛️ Quantum Circuit

The quantum forecasting layer uses a 6-qubit variational circuit. Each weather feature is encoded into the circuit through `AngleEmbedding`, followed by 3 layers of `StronglyEntanglingLayers` to learn nonlinear and entangled feature interactions. The circuit returns PauliZ expectation values, which are passed into the classical neural head for final next-day weather prediction.

Circuit visualizations are saved as:

- `circuit_clean.png`
- `circuit_decomposed.png`
- `quantum_circuit_diagram.png`

---

## 🔄 How It Works

1. Load NASA POWER daily weather data for Dhaka.
2. Scale the six weather features using MinMax normalization.
3. Shift all target variables by one day for next-day forecasting.
4. Train baseline models and the hybrid QML model.
5. Train the SMOTE-enhanced rain classifier.
6. Generate SHAP explanations for regression and rain classification.
7. Serve predictions through a FastAPI webhook connected to Facebook Messenger.
8. Use Groq LLM to convert model outputs into bilingual, user-friendly responses.

---

## 📊 Results

Detailed per-variable results are stored in `full_evaluation_results.csv`. Average model-level summaries are stored in `model_comparison_summary.csv`.

| Model | MAE (avg) | RMSE (avg) | R² (avg) | MAPE (avg %) |
|---|---:|---:|---:|---:|
| Linear Regression | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` |
| Random Forest | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` |
| MLP | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` |
| LSTM | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` |
| QML (Ours) | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` | See `full_evaluation_results.csv` |

No metric values are hardcoded in this README. Use the generated CSV files for the authoritative numerical results.

---

## 💬 Facebook Messenger Chatbot

QuantWeather-BD includes a chatbot deployment layer for making the forecasting system accessible through Facebook Messenger.

| Component | Details |
|---|---|
| API Framework | FastAPI |
| Messenger Integration | Facebook Messenger Platform webhook |
| LLM | Groq API with LLaMA3-8b-8192 |
| Development Tunnel | ngrok |
| Languages | Bengali + English, auto-detected |
| Current Weather | OpenWeatherMap API |
| Tomorrow Forecast | Hybrid QML model + Rain Classifier |
| Explanation | SHAP-based plain-language summary |
| Historical Context | Past weather trend summary |

### Chatbot Flow

```text
User message
  -> Intent detection
  -> OpenWeatherMap current weather lookup
  -> QML next-day weather prediction
  -> Rain Classifier category + confidence
  -> SHAP explanation
  -> Groq LLM response generation
  -> Facebook Messenger reply
```

---

## 🗂️ Project Structure

```text
QuantWeather-BD/
├── main.py
├── .env.example
├── requirements.txt
├── QuantWeather-BD-Final-v2.ipynb
├── Dataset/
│   └── bangladesh_weather_1981_2026.csv
├── eda_timeseries.png
├── eda_correlation.png
├── convergence_plot.png
├── actual_vs_predicted.png
├── confusion_matrix.png
├── roc_curves.png
├── model_comparison_chart.png
├── training_time.png
├── hybrid_model_diagram.png
├── circuit_clean.png
├── circuit_decomposed.png
├── shap_rf_summary.png
├── shap_rf_bar.png
├── shap_qml_summary.png
├── shap_comparison.png
├── shap_rain_importance.png
├── seasonal_analysis.png
├── residual_analysis.png
├── boxplot.png
├── learning_curves_classical.png
├── rain_confusion_matrix.png
├── rain_smote_confusion.png
├── data_distribution.png
├── quantum_circuit_diagram.png
├── full_evaluation_results.csv
├── model_comparison_summary.csv
└── statistical_tests.csv
```

> Note: `.env` files and trained model binaries (`*.pth`, `*.pkl`) are intentionally excluded from GitHub. Train or download them locally before running inference.

---

## 📁 Saved Outputs

### Graphs

| Preview | File | Description |
|---|---|---|
| <img src="eda_timeseries.png" width="220" alt="Weather variables over 45 years"> | `eda_timeseries.png` | Weather variables over 45 years |
| <img src="eda_correlation.png" width="220" alt="Feature correlation heatmap and monthly temperature"> | `eda_correlation.png` | Feature correlation heatmap + monthly temperature |
| <img src="convergence_plot.png" width="220" alt="Training loss for MLP, LSTM, and QML"> | `convergence_plot.png` | Training loss for MLP, LSTM, QML |
| <img src="actual_vs_predicted.png" width="220" alt="Predicted vs actual values"> | `actual_vs_predicted.png` | Predicted vs actual values |
| <img src="confusion_matrix.png" width="220" alt="Rain classifier confusion matrix"> | `confusion_matrix.png` | Rain classifier confusion matrix |
| <img src="roc_curves.png" width="220" alt="ROC curves for rain classifier"> | `roc_curves.png` | ROC curves for rain classifier |
| <img src="model_comparison_chart.png" width="220" alt="Visual comparison of all models"> | `model_comparison_chart.png` | Visual comparison of all models |
| <img src="training_time.png" width="220" alt="Training time comparison"> | `training_time.png` | Training time comparison |
| <img src="hybrid_model_diagram.png" width="220" alt="Full hybrid QML architecture diagram"> | `hybrid_model_diagram.png` | Full hybrid QML architecture diagram |
| <img src="circuit_clean.png" width="220" alt="Variational quantum circuit diagram"> | `circuit_clean.png` | Variational quantum circuit diagram |
| <img src="circuit_decomposed.png" width="220" alt="Decomposed quantum circuit using RZ, RY, and CNOT gates"> | `circuit_decomposed.png` | Decomposed quantum circuit using RZ, RY, and CNOT gates |
| <img src="shap_rf_summary.png" width="220" alt="SHAP summary plot for Random Forest"> | `shap_rf_summary.png` | SHAP summary plot for Random Forest |
| <img src="shap_rf_bar.png" width="220" alt="SHAP feature importance bar plot for Random Forest"> | `shap_rf_bar.png` | SHAP feature importance bar plot for Random Forest |
| <img src="shap_qml_summary.png" width="220" alt="SHAP summary plot for QML"> | `shap_qml_summary.png` | SHAP summary plot for QML |
| <img src="shap_comparison.png" width="220" alt="SHAP comparison between Random Forest and QML"> | `shap_comparison.png` | SHAP comparison between RF and QML |
| <img src="shap_rain_importance.png" width="220" alt="SHAP importance for Rain Classifier"> | `shap_rain_importance.png` | SHAP importance for Rain Classifier |
| <img src="seasonal_analysis.png" width="220" alt="Seasonal weather patterns"> | `seasonal_analysis.png` | Seasonal weather patterns |
| <img src="residual_analysis.png" width="220" alt="Residual analysis plot"> | `residual_analysis.png` | Residual analysis plot |
| <img src="boxplot.png" width="220" alt="Feature distribution boxplots"> | `boxplot.png` | Feature distribution boxplots |
| <img src="learning_curves_classical.png" width="220" alt="Learning curves"> | `learning_curves_classical.png` | Learning curves |
| <img src="rain_confusion_matrix.png" width="220" alt="Rain confusion matrix"> | `rain_confusion_matrix.png` | Rain confusion matrix |
| <img src="rain_smote_confusion.png" width="220" alt="SMOTE rain confusion matrix"> | `rain_smote_confusion.png` | SMOTE rain confusion matrix |
| <img src="data_distribution.png" width="220" alt="Data distribution"> | `data_distribution.png` | Data distribution |
| <img src="quantum_circuit_diagram.png" width="220" alt="Quantum circuit visualization"> | `quantum_circuit_diagram.png` | Quantum circuit visualization |

### CSV Files

| File | Description |
|---|---|
| `full_evaluation_results.csv` | Complete per-variable evaluation |
| `model_comparison_summary.csv` | Average metrics across all models |
| `statistical_tests.csv` | Statistical significance tests |

### Local Model Artifacts

| File | Description |
|---|---|
| `qml_model_final.pth` | Trained QML model weights |
| `mlp_model_final.pth` | Trained MLP model weights |
| `lstm_model_final.pth` | Trained LSTM model weights |
| `lr_model.pkl` | Linear Regression model |
| `rf_model.pkl` | Random Forest model |
| `rain_classifier_smote.pkl` | Rain classifier |
| `scaler_X.pkl` | Input feature scaler |
| `scaler_y.pkl` | Output target scaler |

These files are generated or loaded locally and are ignored by Git because they can be large and may vary between runs.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![PennyLane](https://img.shields.io/badge/PennyLane-QML-674ea7.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688.svg)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-f55036.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-f7931e.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-1f77b4.svg)
![OpenWeatherMap](https://img.shields.io/badge/OpenWeatherMap-Weather-eb6e4b.svg)
![Facebook Messenger](https://img.shields.io/badge/Facebook%20Messenger-Webhook-0866ff.svg)
![ngrok](https://img.shields.io/badge/ngrok-Tunnel-1f1e37.svg)

---

## 🚀 Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create environment variables

Copy `.env.example` to `.env`, then add your private credentials:

```env
VERIFY_TOKEN=your_facebook_verify_token
PAGE_ACCESS_TOKEN=your_facebook_page_access_token
GROQ_API_KEY=your_groq_api_key
OpenWeatherMapAPI=your_openweathermap_api_key
```

### 3. Run the FastAPI server

```bash
uvicorn main:app --port 8000 --reload
```

### 4. Start ngrok

```bash
ngrok http 8000
```

### 5. Configure Facebook webhook

Use the generated ngrok HTTPS URL as the webhook callback URL in the Facebook Developer Console, then set the verify token to match `VERIFY_TOKEN`.

---

## 🔬 Research Notes

QuantWeather-BD is designed as a reproducible research prototype for exploring whether hybrid quantum-classical architectures can provide useful weather forecasting behavior on real-world meteorological data. The project keeps model comparison, statistical testing, explainability, and deployment artifacts together so the full forecasting pipeline can be inspected end to end.

---

Built with ❤️ using Quantum Machine Learning for Bangladesh 🇧🇩
