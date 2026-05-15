# QuantWeather-BD 🌦️⚛️
### Quantum Machine Learning-Based Weather Forecasting for Bangladesh

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.29.1-green.svg)](https://pennylane.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**QuantWeather-BD** is a hybrid Quantum Machine Learning (QML) system for next-day multi-variable weather forecasting in Bangladesh. It combines a **Variational Quantum Circuit (VQC)** with a classical neural network, trained on 45 years of NASA POWER data, and evaluated against four classical baseline models.

---

## 🎯 Objectives

- Predict next-day weather variables using a hybrid QML model
- Compare QML performance against classical ML baselines
- Demonstrate real-world applicability via Telegram Bot integration

---

## 📂 Project Structure

```
QuantWeather-BD/
├── Dataset/
│   └── bangladesh_weather_1981_2026.csv
├── QuantWeather.ipynb               # Main notebook (Colab)
├── qml_model_final.pth              # Trained QML model
├── mlp_model_final.pth              # Trained MLP model
├── lstm_model_final.pth             # Trained LSTM model
├── lr_model.pkl                     # Trained Linear Regression
├── rf_model.pkl                     # Trained Random Forest
├── scaler_X.pkl                     # Feature scaler
├── scaler_y.pkl                     # Target scaler
├── full_evaluation_results.csv      # Per-variable results
├── model_comparison_summary.csv     # Summary comparison
├── statistical_tests.csv            # Wilcoxon test results
├── eda_timeseries.png
├── eda_correlation.png
├── convergence_plot.png
├── actual_vs_predicted.png
├── confusion_matrix.png
├── roc_curves.png
├── model_comparison_chart.png
└── training_time.png
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | NASA POWER (MERRA-2) |
| **Location** | Dhaka, Bangladesh (23.8103°N, 90.4125°E) |
| **Date Range** | January 1, 1981 → May 14, 2026 |
| **Total Records** | 16,570 daily observations |
| **Features** | T2M, T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, WS2M |

### Feature Description

| Feature | Description | Unit |
|---|---|---|
| T2M | Average temperature at 2m | °C |
| T2M_MAX | Maximum temperature at 2m | °C |
| T2M_MIN | Minimum temperature at 2m | °C |
| RH2M | Relative humidity at 2m | % |
| PRECTOTCORR | Corrected precipitation | mm/day |
| WS2M | Wind speed at 2m | m/s |

---

## ⚛️ Model Architecture

### Hybrid QML Model

```
Input (6 features)
       ↓
Quantum Circuit (6 qubits, 3 StronglyEntanglingLayers)
  - AngleEmbedding
  - StronglyEntanglingLayers
  - PauliZ measurements
       ↓
Classical Layers
  - Linear(6 → 16) + ReLU
  - Linear(16 → 6)
       ↓
Output (6 weather variables)
```

| Config | Value |
|---|---|
| Qubits | 6 |
| VQC Layers | 3 |
| Total Parameters | 268 |
| Framework | PennyLane + PyTorch |

---

## 🏋️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.005) |
| Loss Function | MSELoss |
| Epochs | 100 |
| Batch Size | 64 |
| LR Scheduler | StepLR (step=30, γ=0.5) |
| Train/Test Split | 80% / 20% |

---

## 📈 Results

### Model Comparison (Average across 6 variables)

| Model | MAE | RMSE | R² | MAPE (%) |
|---|---|---|---|---|
| Linear Regression | 1.7904 | 3.1333 | 0.7878 | 110.78 |
| Random Forest | 1.8317 | 3.1599 | 0.7839 | 105.19 |
| MLP | 1.8191 | 3.0333 | 0.8001 | 156.22 |
| LSTM | 1.7915 | 3.0196 | 0.8016 | 157.66 |
| **QML (Ours)** | **1.8589** | **3.0628** | **0.7970** | **190.90** |

### Temperature Classification Results (QML)

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| Cold (<20°C) | 0.90 | 0.94 | 0.92 |
| Mild (20-27°C) | 0.89 | 0.85 | 0.87 |
| Hot (27-32°C) | 0.95 | 0.96 | 0.96 |
| Extreme (>32°C) | 0.76 | 0.37 | 0.50 |
| **Overall Accuracy** | | | **92%** |

### Statistical Significance (Wilcoxon Signed-Rank Test)

| Comparison | p-value | Result |
|---|---|---|
| QML vs Linear Regression | 0.004393 | ✅ Significant |
| QML vs Random Forest | 0.029319 | ✅ Significant |
| QML vs MLP | 0.696380 | ➖ Comparable |
| QML vs LSTM | 0.000000 | ✅ Significant |

### Training Time Comparison

| Model | Training Time |
|---|---|
| Linear Regression | 0.1s |
| Random Forest | 10.8s |
| MLP | 41.5s |
| LSTM | 116.3s |
| QML (Ours) | 18,030.7s (~5 hrs) |

---

## ⚙️ Installation

```bash
# Create conda environment
conda create -n quantum python=3.10
conda activate quantum

# Install dependencies
pip install pennylane pennylane-lightning
pip install torch scikit-learn pandas numpy
pip install matplotlib seaborn joblib tqdm scipy
```

---

## 🚀 Usage

### Google Colab (Recommended)
1. Upload `QuantWeather.ipynb` to Google Colab
2. Set Runtime → T4 GPU
3. Upload dataset to Google Drive at:
   `My Drive/QuantWeather-BD/Dataset/bangladesh_weather_1981_2026.csv`
4. Run all cells

---

## 📦 Dependencies

| Package | Version |
|---|---|
| Python | 3.10+ |
| PennyLane | 0.29.1 |
| PyTorch | 2.x |
| scikit-learn | latest |
| pandas | latest |
| numpy | latest |
| matplotlib | latest |
| seaborn | latest |
| scipy | latest |

---

## 🔮 Future Work

- Telegram Bot integration for real-time forecasting
- Precipitation classification (No Rain / Light / Moderate / Heavy)
- Multi-city expansion (Chittagong, Sylhet, Rajshahi)
- Pressure (PS) feature integration
- Ablation study (4 vs 6 qubits)
- Real quantum hardware deployment (IBM Quantum)

---

## 📄 Citation

```bibtex
@article{quantweather_bd_2026,
  title   = {QuantWeather-BD: Quantum Machine Learning-Based 
             Weather Forecasting for Bangladesh},
  author  = {},
  journal = {},
  year    = {2026}
}
```

---

## 📜 License

This project is licensed under the MIT License.

---

*Dataset source: NASA Prediction Of Worldwide Energy Resources (POWER) — https://power.larc.nasa.gov*
