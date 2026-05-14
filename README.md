# QuantWeather-BD

**QuantWeather-BD** is a hybrid Quantum Machine Learning (QML) project for **next-day temperature prediction in Bangladesh** using the NASA POWER daily weather dataset.

The project combines:
- Classical preprocessing and feature engineering
- A Variational Quantum Circuit (VQC) built with PennyLane
- A PyTorch hybrid model (quantum layer + classical output layer)

## Project Objective

Build and evaluate a reproducible end-to-end pipeline to predict **next day `T2M` (2-meter air temperature)** from historical weather variables.

## Dataset

- **Source:** NASA POWER Daily Weather Data
- **Coverage:** January 1, 1981 to May 14, 2026
- **Location:** Bangladesh region (lat 23.8103, lon 90.4125)
- **Primary file used in notebook:**
  - `Dataset/bangladesh_weather_1981_2026.csv`

### Input Variables

- `T2M` - Temperature at 2 meters (C)
- `T2M_MAX` - Maximum temperature at 2 meters (C)
- `T2M_MIN` - Minimum temperature at 2 meters (C)
- `RH2M` - Relative humidity at 2 meters (%)
- `PRECTOTCORR` - Corrected precipitation (mm/day)
- `WS2M` - Wind speed at 2 meters (m/s)

### Data Notes

- NASA POWER missing value code `-999` is replaced with `NaN`
- Header metadata rows are skipped (`skiprows=16`)
- `DATE` is constructed from `YEAR`, `MO`, and `DY`

## Methodology

The notebook implements the following pipeline:

1. **Data Loading and Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering**
4. **QML Model Construction**
5. **Training**
6. **Evaluation**
7. **Model and Scaler Saving**

### Target Definition

- Target variable: **next day temperature**
- `TARGET_T2M_NEXT_DAY = T2M.shift(-1)`

### Train/Test Strategy

- Chronological split (no shuffling)
- 80% training, 20% testing
- Feature scaling with `MinMaxScaler` to `[0, 1]`

## Quantum Model Design

- **Framework:** PennyLane + PyTorch
- **Device:** `default.qubit`
- **Qubits:** `6`
- **Embedding:** `AngleEmbedding`
- **Variational block:** `StronglyEntanglingLayers`
- **Hybrid head:** Classical `nn.Linear` output layer

## Training Configuration

- Loss: `MSELoss`
- Optimizer: `Adam`
- Learning rate: `0.01`
- Epochs: `100`
- Progress logging every `10` epochs

## Evaluation Metrics

- MAE
- RMSE
- R^2 score
- Actual vs Predicted temperature plot

## Project Outputs

After running the notebook, the following artifacts are generated:

- `quantweather_bd_model.pth` - Trained PyTorch hybrid QML model weights
- `scaler.pkl` - Fitted `MinMaxScaler`

## Repository Structure

```text
Quantum-Enhanced Weather Forecasting for Bangladesh/
|-- Dataset/
|   |-- bangladesh_weather_1981_2026.csv
|-- QuantWeather-BD.ipynb
|-- quantweather_bd_model.pth
|-- scaler.pkl
|-- README.md
```

## Setup Instructions

### 1. Create and activate environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install pennylane torch pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `QuantWeather-BD.ipynb` and run cells sequentially.

## Reproducibility

The notebook sets random seeds for NumPy and PyTorch to improve run-to-run consistency.

## Future Improvements

- Multi-step forecasting (2 to 7 day horizon)
- Additional weather and seasonal lag features
- Hyperparameter tuning for circuit depth and optimizer
- Comparison against classical baselines (LSTM, XGBoost, Random Forest)
- Station-wise or region-wise modeling across Bangladesh

## License

This project is open for academic and research use. Add a formal license file (`LICENSE`) if you plan to publish or distribute.

## Author

**QuantWeather-BD Project**

If you want, this README can be upgraded next with badges, benchmark tables, and publication-style result reporting.
