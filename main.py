"""FastAPI service for plant disease detection and FloodGuard sensor data."""

import os
from io import BytesIO

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from supabase import create_client

from tensorflow.keras.models import load_model
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Layer,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "plant_disease_mobilenet.h5"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "healthy",
    "leaf_blast",
    "leaf_blight",
]


# ============================================================
# SUPABASE CONFIGURATION
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )
        print("Supabase connected successfully.")
    except Exception as exc:
        print("Supabase connection failed:", exc)
        supabase = None
else:
    print("Supabase environment variables not configured.")


# ============================================================
# KERAS COMPATIBILITY LAYERS
# ============================================================

class TrueDivide(Layer):
    """Compatibility shim for older MobileNetV2 H5 files."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, divisor):
        return inputs / divisor


class Subtract(Layer):
    """Compatibility shim for older MobileNetV2 H5 files."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, value):
        return inputs - value


class CompatibleDense(Dense):
    """Compatibility shim for Dense layers saved with older Keras."""

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return super().from_config(config)


# ============================================================
# LOAD PLANT DISEASE MODEL
# ============================================================

def load_compatible_model():

    try:

        return load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={
                "TrueDivide": TrueDivide,
                "Subtract": Subtract,
                "Dense": CompatibleDense,
            },
        )

    except Exception:

        print(
            "Standard model loading failed."
        )

        print(
            "Trying compatibility loading..."
        )

        inputs = Input(
            shape=IMAGE_SIZE + (3,)
        )

        base = MobileNetV2(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            weights=None,
        )

        x = base(
            inputs,
            training=False
        )

        x = GlobalAveragePooling2D()(x)

        x = Dropout(0.2)(x)

        outputs = Dense(
            len(CLASS_NAMES),
            activation="softmax"
        )(x)

        compatible_model = Model(
            inputs,
            outputs
        )

        compatible_model.load_weights(
            MODEL_PATH
        )

        return compatible_model


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="FloodGuard + Plant Disease Detection API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================

latest_sensor_packet = {}

try:

    model = load_compatible_model()

    MODEL_LOAD_ERROR = None

    print(
        "Plant disease model loaded successfully."
    )

except Exception as exc:

    model = None

    MODEL_LOAD_ERROR = str(exc)

    print(
        "Plant disease model failed to load:"
    )

    print(
        MODEL_LOAD_ERROR
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(contents: bytes) -> np.ndarray:
    """
    Decode uploaded image and prepare it
    for MobileNetV2.
    """

    try:

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail="Upload a valid image file."
        ) from exc

    image = image.resize(
        IMAGE_SIZE
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing
    array = (
        array / 127.5
    ) - 1.0

    return np.expand_dims(
        array,
        axis=0
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health() -> dict:

    return {
        "status": (
            "ok"
            if model is not None
            else "model_not_loaded"
        ),
        "plant_disease_model": (
            "loaded"
            if model is not None
            else "not_loaded"
        ),
        "supabase": (
            "connected"
            if supabase is not None
            else "not_configured"
        ),
    }


# ============================================================
# SENSOR INGEST
# ESP32 → FASTAPI → SUPABASE
# ============================================================

@app.post("/sensor-ingest")
async def sensor_ingest(packet: dict) -> dict:

    # --------------------------------------------------------
    # REQUIRED SENSOR FIELDS
    # --------------------------------------------------------

    required_fields = {
        "temperature",
        "humidity",
        "latitude",
        "longitude",
    }

    missing_fields = sorted(
        required_fields - packet.keys()
    )

    if missing_fields:

        raise HTTPException(
            status_code=400,
            detail=(
                "Missing sensor fields: "
                + ", ".join(missing_fields)
            ),
        )


    # --------------------------------------------------------
    # RAIN LEVEL
    # Supports:
    # rain_level
    # rainfall
    # --------------------------------------------------------

    rain_level = packet.get(
        "rain_level",
        packet.get("rainfall")
    )

    if rain_level is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Missing sensor field: "
                "rain_level or rainfall"
            ),
        )


    # --------------------------------------------------------
    # WATER LEVEL
    # --------------------------------------------------------

    water_level = packet.get(
        "water_level"
    )

    if water_level is None:

        raise HTTPException(
            status_code=400,
            detail="Missing sensor field: water_level"
        )


    # --------------------------------------------------------
    # GET SENSOR VALUES
    # --------------------------------------------------------

    try:

        temperature = float(
            packet["temperature"]
        )

        humidity = float(
            packet["humidity"]
        )

        rain_level = int(
            float(rain_level)
        )

        water_level = float(
            water_level
        )

        latitude = float(
            packet["latitude"]
        )

        longitude = float(
            packet["longitude"]
        )

    except (ValueError, TypeError) as exc:

        raise HTTPException(
            status_code=400,
            detail="Sensor values must be numeric."
        ) from exc


    # --------------------------------------------------------
    # CALCULATE STATUS
    # --------------------------------------------------------

    if (
        water_level >= 80
        or rain_level >= 2500
    ):

        status = "DANGER"

    elif (
        water_level >= 50
        or rain_level >= 1500
    ):

        status = "WARNING"

    else:

        status = "SAFE"


    # --------------------------------------------------------
    # SAVE LATEST PACKET IN MEMORY
    # --------------------------------------------------------

    latest_sensor_packet.clear()

    latest_sensor_packet.update(
        {
            "temperature": temperature,
            "humidity": humidity,
            "rain_level": rain_level,
            "water_level": water_level,
            "latitude": latitude,
            "longitude": longitude,
            "status": status,
            "nodeId": str(
                packet.get(
                    "nodeId",
                    "FG_NODE_01"
                )
            ),
        }
    )


    # --------------------------------------------------------
    # SAVE TO SUPABASE
    # --------------------------------------------------------

    if supabase is None:

        return {
            "status": "accepted",
            "supabase": "not_configured",
            "message": (
                "Sensor received but "
                "Supabase is not configured."
            ),
            "sensor_data": latest_sensor_packet,
        }


    # IMPORTANT:
    # "temparature" is intentionally spelled this way
    # because that is the column name in your existing
    # Supabase table.

    record = {

        "temparature": temperature,

        "humidity": humidity,

        "rain_level": rain_level,

        "water_level": water_level,

        "latitude": latitude,

        "longitude": longitude,

        "status": status,
    }


    try:

        result = (
            supabase
            .table("sensor_data")
            .insert(record)
            .execute()
        )

        return {

            "status": "accepted",

            "supabase": "stored",

            "nodeId": str(
                packet.get(
                    "nodeId",
                    "FG_NODE_01"
                )
            ),

            "sensor_data": latest_sensor_packet,

            "database_record": result.data,
        }

    except Exception as exc:

        print(
            "Supabase insert error:"
        )

        print(exc)

        return {

            "status": "accepted",

            "supabase": "error",

            "message": str(exc),

            "sensor_data": latest_sensor_packet,
        }


# ============================================================
# LATEST SENSOR DATA
# ============================================================

@app.get("/sensor-ingest/latest")
def latest_sensor_ingest() -> dict:

    # Try Supabase first
    if supabase is not None:

        try:

            result = (
                supabase
                .table("sensor_data")
                .select("*")
                .order(
                    "id",
                    desc=True
                )
                .limit(1)
                .execute()
            )

            if result.data:

                return result.data[0]

        except Exception as exc:

            print(
                "Supabase latest-data error:",
                exc
            )


    # Fallback to memory
    return latest_sensor_packet


# ============================================================
# LATEST SENSOR DATA
# Simple endpoint for frontend
# ============================================================

@app.get("/sensor-data")
def get_sensor_data() -> dict:

    if supabase is not None:

        try:

            result = (
                supabase
                .table("sensor_data")
                .select("*")
                .order(
                    "id",
                    desc=True
                )
                .limit(1)
                .execute()
            )

            if result.data:

                return result.data[0]

        except Exception as exc:

            print(
                "Sensor data error:",
                exc
            )

    return latest_sensor_packet


# ============================================================
# SENSOR HISTORY
# ============================================================

@app.get("/sensor-history")
def get_sensor_history() -> list:

    if supabase is None:

        return []

    try:

        result = (
            supabase
            .table("sensor_data")
            .select("*")
            .order(
                "id",
                desc=True
            )
            .limit(100)
            .execute()
        )

        return result.data

    except Exception as exc:

        print(
            "Sensor history error:",
            exc
        )

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve sensor history."
        )


# ============================================================
# PLANT DISEASE PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
) -> dict:

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model could not be loaded: "
                + str(MODEL_LOAD_ERROR)
            ),
        )


    # Read uploaded image
    contents = await file.read()


    # Prepare image
    image = prepare_image(
        contents
    )


    # Prediction
    probabilities = model.predict(
        image,
        verbose=0
    )[0]


    # Highest probability class
    class_index = int(
        np.argmax(probabilities)
    )


    # Check class names
    if class_index >= len(
        CLASS_NAMES
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "CLASS_NAMES does not match "
                "the trained model output."
            ),
        )


    return {

        "disease":
            CLASS_NAMES[class_index],

        "confidence":
            float(
                probabilities[class_index]
            ),
    }


@app.post("/expert-chat")
async def expert_chat(payload: dict) -> dict:
    """Return a live, context-aware agronomy response for the expert chat."""
    query = str(payload.get("query", "")).strip()
    if not query:
        raise HTTPException(status_code=400, detail="A farmer question is required.")

    crop = str(payload.get("crop", "the selected crop"))
    stage = str(payload.get("stage", "current stage"))
    district = str(payload.get("district", "your district"))
    risk_score = float(payload.get("riskScore", 0))
    sensors = payload.get("sensors", {})
    weather = payload.get("weather") or {}
    lower_query = query.lower()
    actions = []
    sources = ["crop profile", "ESP32 telemetry"]

    asks_weather = any(word in lower_query for word in ("rain", "weather", "spray", "irrigat", "water"))
    asks_disease = any(word in lower_query for word in ("disease", "symptom", "spot", "blast", "pest", "insect", "leaf"))
    asks_market = any(word in lower_query for word in ("price", "sell", "mandi", "market", "rate"))
    asks_nutrition = any(word in lower_query for word in ("fertil", "nutrient", "manure", "compost", "urea", "feed"))
    asks_calendar = any(word in lower_query for word in ("when", "stage", "calendar", "harvest", "sow", "sowing", "task", "plan", "next step"))

    if asks_weather:
        sources.append("live weather")
        rain = weather.get("current", {}).get("rain", sensors.get("rainfall", "unavailable"))
        actions.append(f"Check the next rain window before field work; current rain signal is {rain} mm.")
    if asks_disease:
        sources.append("disease evidence")
        actions.append("Scout five plants in each field zone and upload a close leaf image if symptoms are visible.")
    if asks_market:
        sources.append("mandi reference")
        actions.append("Verify today's local APMC price before selling or committing to a buyer.")
    if asks_nutrition:
        sources.append("soil and nutrient guidance")
        actions.append("Use a soil test before adding fertilizer; prefer a crop-stage compost or biofertilizer plan over a blanket dose.")
    if asks_calendar:
        sources.append("crop calendar")
        actions.append(f"Review the {stage} task list for {crop} and record the next field observation after completing it.")

    humidity = sensors.get("humidity", "unknown")
    temperature = sensors.get("temperature", "unknown")
    rain = weather.get("current", {}).get("rain", sensors.get("rainfall", "unavailable"))
    if asks_disease:
        text = (
            f"Disease check for {crop}: at {stage}, the field risk score is {risk_score:.0f}% with "
            f"{humidity}% humidity and {temperature}°C. Do not spray from symptoms alone; isolate a sample, "
            "inspect five plants per zone, and confirm the disease before treatment."
        )
    elif asks_weather:
        text = (
            f"Weather plan for {crop} in {district}: the current rain signal is {rain} mm, "
            f"temperature is {temperature}°C, and humidity is {humidity}%. Delay foliar spraying before rain; "
            "irrigate only when the root zone is below the crop threshold."
        )
    elif asks_market:
        text = (
            f"Selling plan for {crop} in {district}: compare at least two local mandi quotes, "
            "check quality deductions and transport cost, then record the final rate before selling. "
            f"The current field is at the {stage} stage, so avoid harvesting early only for a short price movement."
        )
    elif asks_nutrition:
        text = (
            f"Nutrition plan for {crop} at {stage}: first check soil moisture ({sensors.get('soilMoisture', 'unknown')} ADC) "
            "and use a soil test. Split organic or biofertilizer inputs around the crop stage instead of applying a blanket dose."
        )
    elif asks_calendar:
        text = (
            f"Calendar update for {crop}: the field is marked at {stage} in {district}. "
            "Complete the stage tasks, scout before inputs, and set a follow-up observation for the next 24 to 48 hours."
        )
    else:
        actions.append("Record the observation, compare it with the crop threshold, and review again after 24 hours.")
        text = (
            f"For {crop} at the {stage} stage in {district}, field risk is {risk_score:.0f}%. "
            f"The node reports {temperature}°C and {humidity}% humidity. Start by recording the field observation "
            "and compare it with the crop profile before changing inputs."
        )
    return {"text": text, "actions": actions, "dataSources": sources}