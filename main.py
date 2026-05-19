import logging
import os
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pennylane as qml
import shap
import torch
import torch.nn as nn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from sklearn.exceptions import InconsistentVersionWarning

load_dotenv()
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantweather")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("shap").setLevel(logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OpenWeatherMapAPI")

FACEBOOK_MESSAGES_URL = "https://graph.facebook.com/me/messages"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"
OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

FEATURE_NAMES = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M"]
RAIN_CLASSES_EN = ["No Rain", "Light", "Moderate", "Heavy"]
RAIN_CLASSES_BN = ["বৃষ্টি নেই", "হালকা", "মাঝারি", "ভারী"]

SYSTEM_PROMPT = """
You are QuantWeather-BD, an AI-powered bilingual weather assistant for Bangladesh built using Quantum Machine Learning.

LANGUAGE RULE (STRICT):
- User writes in Bengali → respond ONLY in Bengali
- User writes in English → respond ONLY in English

You will receive weather data as context. Format it beautifully with emojis and explain in simple language.

For tomorrow prediction, explain WHY the model gave this result using the SHAP top features — in plain simple language, NO raw numbers or technical terms.

Use the exact values from the context. Do not convert units. Do not invent advice, trends, or labels.
For Bengali, use Bengali labels and simple natural Bengali. For English, use English labels only.
Do not include a date unless the context provides one.
For tomorrow prediction, include only: temperature, max, min, humidity, rain category with confidence, wind speed, and a short reason.

Never make up weather data. Only use what is given in the context.
""".strip()


class HybridQMLModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantum = build_quantum_layer()
        self.fc1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.quantum(x)
        out = self.relu(self.fc1(q_out))
        return self.fc2(out)


class MLPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ModelStore:
    qml_model: HybridQMLModel
    mlp_model: MLPModel
    rain_classifier: Any
    rf_model: Any
    lr_model: Any
    scaler_x: Any
    scaler_y: Any


models: ModelStore | None = None


def build_quantum_layer() -> qml.qnn.TorchLayer:
    n_qubits = 6
    n_layers = 3
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="torch")
    def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[Any]:
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return qml.qnn.TorchLayer(
        quantum_circuit,
        {"weights": (n_layers, n_qubits, 3)},
    )


def load_models() -> ModelStore:
    qml_model = HybridQMLModel()
    qml_model.load_state_dict(torch.load(BASE_DIR / "qml_model_final.pth", map_location="cpu"))
    qml_model.eval()

    mlp_model = MLPModel()
    mlp_model.load_state_dict(torch.load(BASE_DIR / "mlp_model_final.pth", map_location="cpu"))
    mlp_model.eval()

    return ModelStore(
        qml_model=qml_model,
        mlp_model=mlp_model,
        rain_classifier=joblib.load(BASE_DIR / "rain_classifier_smote.pkl"),
        rf_model=joblib.load(BASE_DIR / "rf_model.pkl"),
        lr_model=joblib.load(BASE_DIR / "lr_model.pkl"),
        scaler_x=joblib.load(BASE_DIR / "scaler_X.pkl"),
        scaler_y=joblib.load(BASE_DIR / "scaler_y.pkl"),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global models
    try:
        models = load_models()
        logger.info("All ML models loaded successfully")
    except Exception:
        models = None
        logger.exception("Failed to load one or more ML models")
    yield


app = FastAPI(title="QuantWeather-BD Messenger Bot", lifespan=lifespan)


@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if VERIFY_TOKEN and mode == "subscribe" and token == VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge)

    return PlainTextResponse("Invalid verification token", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        logger.exception("Invalid Facebook webhook payload")
        return {"status": "ignored"}

    if payload.get("object") != "page":
        return {"status": "ignored"}

    for entry in payload.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event.get("sender", {}).get("id")
            message = event.get("message", {})
            text = message.get("text")

            if not sender_id or not text or message.get("is_echo"):
                continue

            try:
                reply = await handle_user_message(text)
                await send_facebook_message(sender_id, reply)
            except Exception:
                logger.exception("Failed to handle Messenger event")
                await send_facebook_message(sender_id, fallback_message(detect_language(text)))

    return {"status": "ok"}


async def handle_user_message(user_message: str) -> str:
    language = detect_language(user_message)
    intent = detect_intent(user_message)

    if intent == "current":
        context = await build_current_weather_context(language)
    elif intent == "tomorrow":
        context = await build_tomorrow_prediction_context(language)
    elif intent == "past":
        context = await build_past_weather_context(language)
    else:
        context = build_general_context(language)

    return await ask_groq(user_message, context, language)


def detect_language(text: str) -> str:
    return "bn" if any("\u0980" <= char <= "\u09ff" for char in text) else "en"


def detect_intent(text: str) -> str:
    lowered = text.lower()

    tomorrow_keywords = ["কাল", "tomorrow", "পূর্বাভাস", "forecast"]
    past_keywords = ["গত", "আগের", "last few days", "past"]
    current_keywords = ["current", "এখন", "আজকে", "আবহাওয়া কেমন", "আবহাওয়া কেমন"]

    if any(keyword in lowered for keyword in past_keywords):
        return "past"
    if any(keyword in lowered for keyword in tomorrow_keywords):
        return "tomorrow"
    if any(keyword in lowered for keyword in current_keywords):
        return "current"

    return "general"


async def build_current_weather_context(language: str) -> str:
    weather = await fetch_current_weather()
    if not weather:
        return unavailable_context("current weather", language)

    features = features_from_current_weather(weather)
    return "\n".join(
        [
            f"LANGUAGE: {language}",
            "INTENT: current_weather",
            "LOCATION: Dhaka, Bangladesh",
            f"TEMPERATURE_C: {features['T2M']:.1f}",
            f"FEELS_LIKE_C: {weather.get('main', {}).get('feels_like', 0):.1f}",
            f"HUMIDITY_PERCENT: {features['RH2M']:.0f}",
            f"RAINFALL_MM: {features['PRECTOTCORR']:.1f}",
            f"WIND_SPEED_MPS: {features['WS2M']:.1f}",
            "INSTRUCTION: Format as current weather only. Do not mention prediction.",
        ]
    )


async def build_tomorrow_prediction_context(language: str) -> str:
    if models is None:
        return unavailable_context("prediction model", language)

    weather = await fetch_current_weather()
    if not weather:
        return unavailable_context("current weather", language)

    forecast = await fetch_forecast()
    features = features_from_current_weather(weather)

    try:
        prediction = run_qml_prediction(features)
        rain_class, rain_confidence = run_rain_classifier(features)
        shap_features = run_qml_shap(features)
    except Exception:
        logger.exception("Tomorrow prediction failed")
        return unavailable_context("tomorrow prediction", language)

    rain_label = RAIN_CLASSES_BN[rain_class] if language == "bn" else RAIN_CLASSES_EN[rain_class]
    top_features = ", ".join(shap_features)
    explanation_hints = build_explanation_hints(shap_features, language)
    forecast_points = len(forecast.get("list", [])) if forecast else 0

    return "\n".join(
        [
            f"LANGUAGE: {language}",
            "INTENT: tomorrow_prediction",
            "LOCATION: Dhaka, Bangladesh",
            "SOURCE_INPUT: today's OpenWeatherMap current data",
            f"FORECAST_API_POINTS_FETCHED: {forecast_points}",
            f"INPUT_TEMPERATURE_C: {features['T2M']:.1f}",
            f"INPUT_MAX_TEMPERATURE_C: {features['T2M_MAX']:.1f}",
            f"INPUT_MIN_TEMPERATURE_C: {features['T2M_MIN']:.1f}",
            f"INPUT_HUMIDITY_PERCENT: {features['RH2M']:.0f}",
            f"INPUT_RAINFALL_MM: {features['PRECTOTCORR']:.1f}",
            f"INPUT_WIND_SPEED_MPS: {features['WS2M']:.1f}",
            f"PREDICTED_TEMPERATURE_C: {prediction['T2M']:.1f}",
            f"PREDICTED_MAX_TEMPERATURE_C: {prediction['T2M_MAX']:.1f}",
            f"PREDICTED_MIN_TEMPERATURE_C: {prediction['T2M_MIN']:.1f}",
            f"PREDICTED_HUMIDITY_PERCENT: {prediction['RH2M']:.0f}",
            f"PREDICTED_WIND_SPEED_MPS: {prediction['WS2M']:.1f}",
            f"RAIN_CATEGORY: {rain_label}",
            f"RAIN_CONFIDENCE_PERCENT: {rain_confidence * 100:.0f}",
            "EXPLAINER: SHAP KernelExplainer on QML model temperature output",
            f"SHAP_TOP_FEATURES: {top_features}",
            f"EXPLANATION_HINTS: {explanation_hints}",
            "INSTRUCTION: Format exactly as a concise tomorrow forecast with emojis. Use only the predicted values. Explain why using EXPLANATION_HINTS. Do not mention raw SHAP, feature codes, or charts.",
        ]
    )


async def build_past_weather_context(language: str) -> str:
    forecast = await fetch_forecast()
    if not forecast:
        return unavailable_context("past weather", language)

    rows = forecast.get("list", [])[:8]
    if not rows:
        return unavailable_context("past weather", language)

    temps = [row.get("main", {}).get("temp") for row in rows if row.get("main", {}).get("temp") is not None]
    humidities = [
        row.get("main", {}).get("humidity")
        for row in rows
        if row.get("main", {}).get("humidity") is not None
    ]
    winds = [row.get("wind", {}).get("speed") for row in rows if row.get("wind", {}).get("speed") is not None]
    rainfall = sum(get_rainfall_mm(row) for row in rows)

    if not temps or not humidities or not winds:
        return unavailable_context("past weather", language)

    return "\n".join(
        [
            f"LANGUAGE: {language}",
            "INTENT: past_weather_trend",
            "LOCATION: Dhaka, Bangladesh",
            "NOTE: OpenWeatherMap free forecast endpoint was used because the requested endpoint is forecast data.",
            f"AVERAGE_TEMPERATURE_C: {np.mean(temps):.1f}",
            f"MIN_TEMPERATURE_C: {np.min(temps):.1f}",
            f"MAX_TEMPERATURE_C: {np.max(temps):.1f}",
            f"AVERAGE_HUMIDITY_PERCENT: {np.mean(humidities):.0f}",
            f"TOTAL_RAINFALL_MM: {rainfall:.1f}",
            f"AVERAGE_WIND_SPEED_MPS: {np.mean(winds):.1f}",
            "INSTRUCTION: Summarize the recent trend clearly and honestly. Do not claim true historical data if only forecast data is available.",
        ]
    )


def build_general_context(language: str) -> str:
    if language == "bn":
        instruction = "ব্যবহারকারীকে বলুন আপনি ঢাকার বর্তমান, কালকের পূর্বাভাস, বা সাম্প্রতিক আবহাওয়ার সারাংশ দিতে পারেন।"
    else:
        instruction = "Tell the user you can provide current Dhaka weather, tomorrow's forecast, or a recent weather summary."

    return "\n".join(
        [
            f"LANGUAGE: {language}",
            "INTENT: general_help",
            "LOCATION: Dhaka, Bangladesh",
            f"INSTRUCTION: {instruction}",
        ]
    )


async def fetch_current_weather() -> dict[str, Any] | None:
    if not OPENWEATHER_API_KEY:
        logger.error("OpenWeatherMap API key is missing")
        return None

    params = {
        "q": "Dhaka,BD",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }
    return await get_openweather(OPENWEATHER_CURRENT_URL, params)


async def fetch_forecast() -> dict[str, Any] | None:
    if not OPENWEATHER_API_KEY:
        logger.error("OpenWeatherMap API key is missing")
        return None

    params = {
        "q": "Dhaka,BD",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "cnt": 40,
    }
    return await get_openweather(OPENWEATHER_FORECAST_URL, params)


async def get_openweather(url: str, params: dict[str, Any]) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("OpenWeatherMap error %s: %s", exc.response.status_code, exc.response.text)
    except httpx.HTTPError:
        logger.exception("OpenWeatherMap request failed")
    return None


def features_from_current_weather(weather: dict[str, Any]) -> dict[str, float]:
    main = weather.get("main", {})
    wind = weather.get("wind", {})
    rain_mm = get_rainfall_mm(weather)

    temp = float(main.get("temp", 0.0))
    return {
        "T2M": temp,
        "T2M_MAX": float(main.get("temp_max", temp)),
        "T2M_MIN": float(main.get("temp_min", temp)),
        "RH2M": float(main.get("humidity", 0.0)),
        "PRECTOTCORR": rain_mm,
        "WS2M": float(wind.get("speed", 0.0)),
    }


def run_qml_prediction(features: dict[str, float]) -> dict[str, float]:
    store = require_models()
    raw_input = feature_array(features)
    scaled_input = store.scaler_x.transform(raw_input)

    with torch.no_grad():
        scaled_prediction = store.qml_model(torch.tensor(scaled_input, dtype=torch.float32)).numpy()

    prediction = store.scaler_y.inverse_transform(scaled_prediction)[0]
    result = dict(zip(FEATURE_NAMES, prediction, strict=True))
    result["RH2M"] = float(np.clip(result["RH2M"], 0, 100))
    result["PRECTOTCORR"] = float(max(result["PRECTOTCORR"], 0))
    result["WS2M"] = float(max(result["WS2M"], 0))
    return {key: float(value) for key, value in result.items()}


def run_rain_classifier(features: dict[str, float]) -> tuple[int, float]:
    store = require_models()
    scaled_input = store.scaler_x.transform(feature_array(features))
    rain_class = int(store.rain_classifier.predict(scaled_input)[0])

    confidence = 1.0
    if hasattr(store.rain_classifier, "predict_proba"):
        probabilities = store.rain_classifier.predict_proba(scaled_input)[0]
        confidence = float(probabilities[rain_class])

    return rain_class, confidence


def run_qml_shap(features: dict[str, float]) -> list[str]:
    store = require_models()
    scaled_input = store.scaler_x.transform(feature_array(features))
    background = build_local_shap_background(scaled_input)

    def predict_t2m(scaled_rows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(scaled_rows, dtype=torch.float32)
            scaled_output = store.qml_model(tensor).numpy()
        output = store.scaler_y.inverse_transform(scaled_output)
        return output[:, 0]

    try:
        explainer = shap.KernelExplainer(predict_t2m, background)
        shap_values = explainer.shap_values(scaled_input, nsamples=50, silent=True)
        importances = np.abs(np.asarray(shap_values)).reshape(-1)
        top_indices = importances.argsort()[-3:][::-1]
        return [FEATURE_NAMES[index] for index in top_indices]
    except Exception:
        logger.exception("SHAP KernelExplainer failed")
        return ["T2M", "RH2M", "PRECTOTCORR"]


def build_local_shap_background(scaled_input: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(42)
    clipped_input = np.clip(scaled_input, 0.0, 1.0)
    local_samples = rng.normal(loc=clipped_input, scale=0.08, size=(20, len(FEATURE_NAMES)))
    local_samples = np.clip(local_samples, 0.0, 1.0)
    local_samples[0] = clipped_input[0]
    return local_samples


def build_explanation_hints(feature_names: list[str], language: str) -> str:
    bn_hints = {
        "T2M": "আজকের গড় তাপমাত্রা কালকের তাপমাত্রার পূর্বাভাসে প্রভাব ফেলেছে",
        "T2M_MAX": "আজকের সর্বোচ্চ তাপমাত্রা গরমের ধারা দেখিয়েছে",
        "T2M_MIN": "আজকের সর্বনিম্ন তাপমাত্রা রাতের আবহাওয়ার ধারা দেখিয়েছে",
        "RH2M": "আজকের আর্দ্রতা বৃষ্টির সম্ভাবনা বুঝতে সাহায্য করেছে",
        "PRECTOTCORR": "সাম্প্রতিক বৃষ্টির পরিমাণ বৃষ্টির ধরন নির্ধারণে প্রভাব ফেলেছে",
        "WS2M": "বাতাসের গতি আবহাওয়ার পরিবর্তনের ইঙ্গিত দিয়েছে",
    }
    en_hints = {
        "T2M": "today's average temperature influenced tomorrow's temperature forecast",
        "T2M_MAX": "today's maximum temperature showed the daytime heat pattern",
        "T2M_MIN": "today's minimum temperature reflected the nighttime weather pattern",
        "RH2M": "today's humidity helped estimate the rain possibility",
        "PRECTOTCORR": "recent rainfall influenced the predicted rain category",
        "WS2M": "wind speed suggested how the weather may change",
    }
    hints = bn_hints if language == "bn" else en_hints
    return "; ".join(hints.get(name, name) for name in feature_names)


async def ask_groq(user_message: str, context: str, language: str) -> str:
    if not GROQ_API_KEY:
        return fallback_message(language)

    language_name = "Bengali" if language == "bn" else "English"
    user_content = (
        f"Detected user language: {language_name}.\n"
        f"You must respond only in {language_name}.\n\n"
        f"User message:\n{user_message}\n\n"
        f"Weather context:\n{context}"
    )
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    response_text = await post_groq(payload)
    if response_text is not None and response_matches_language(response_text, language):
        return response_text

    payload["model"] = GROQ_FALLBACK_MODEL
    response_text = await post_groq(payload)
    if response_text is not None and response_matches_language(response_text, language):
        return response_text

    correction_payload = {
        "model": GROQ_FALLBACK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Rewrite the answer using only {language_name}. "
                    "Do not add any information that is not in the context.\n\n"
                    f"Weather context:\n{context}"
                ),
            },
        ],
    }
    response_text = await post_groq(correction_payload)
    if response_text is not None and response_matches_language(response_text, language):
        return response_text

    return fallback_message(language)


async def post_groq(payload: dict[str, Any]) -> str | None:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content")
    except httpx.HTTPStatusError as exc:
        logger.error("Groq API error %s: %s", exc.response.status_code, exc.response.text)
    except httpx.HTTPError:
        logger.exception("Groq API request failed")
    return None


async def send_facebook_message(recipient_id: str, text: str) -> None:
    if not PAGE_ACCESS_TOKEN:
        logger.error("PAGE_ACCESS_TOKEN is missing")
        return

    params = {"access_token": PAGE_ACCESS_TOKEN}
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text},
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(FACEBOOK_MESSAGES_URL, params=params, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Facebook API error %s: %s", exc.response.status_code, exc.response.text)
    except httpx.HTTPError:
        logger.exception("Facebook message send failed")


def require_models() -> ModelStore:
    if models is None:
        raise RuntimeError("Models are not loaded")
    return models


def feature_array(features: dict[str, float]) -> np.ndarray:
    return np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)


def get_rainfall_mm(weather_row: dict[str, Any]) -> float:
    rain = weather_row.get("rain") or {}
    return float(rain.get("3h") or rain.get("1h") or 0.0)


def response_matches_language(text: str, language: str) -> bool:
    has_bengali = any("\u0980" <= char <= "\u09ff" for char in text)
    if language == "bn":
        return has_bengali
    return not has_bengali


def unavailable_context(resource: str, language: str) -> str:
    return "\n".join(
        [
            f"LANGUAGE: {language}",
            "INTENT: unavailable",
            f"UNAVAILABLE_RESOURCE: {resource}",
            "INSTRUCTION: Apologize briefly and say the requested data is unavailable right now. Do not invent data.",
        ]
    )


def fallback_message(language: str) -> str:
    if language == "bn":
        return "দুঃখিত, এখন উত্তর তৈরি করা যাচ্ছে না। একটু পরে আবার চেষ্টা করুন।"
    return "Sorry, I could not prepare a response right now. Please try again later."
