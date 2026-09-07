"""Run one plant disease prediction from a local image.

Usage:
    python predict.py path/to/plant_image.jpg
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Layer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = Path(__file__).resolve().parent / "plant_disease_mobilenet.h5"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["healthy", "leaf_blast", "leaf_blight"]


class TrueDivide(Layer):
    """Compatibility shim for MobileNetV2 preprocessing saved in older H5 files."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, divisor):
        return inputs / divisor


class Subtract(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, value):
        return inputs - value


class CompatibleDense(Dense):
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return super().from_config(config)


def load_compatible_model():
    try:
        return load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"TrueDivide": TrueDivide, "Subtract": Subtract, "Dense": CompatibleDense},
        )
    except Exception:
        inputs = Input(shape=IMAGE_SIZE + (3,))
        base = MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights=None)
        x = base(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(len(CLASS_NAMES), activation="softmax")(x)
        compatible_model = Model(inputs, outputs)
        compatible_model.load_weights(MODEL_PATH)
        return compatible_model


def load_and_preprocess_image(image_path: str | Path) -> np.ndarray:
    """Load an image and prepare it for the MobileNetV2 model."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)


def predict_image(image_path: str | Path) -> tuple[str, float]:
    """Return the predicted class name and confidence as a decimal."""
    model = load_compatible_model()
    image_batch = load_and_preprocess_image(image_path)
    probabilities = model.predict(image_batch, verbose=0)[0]
    class_index = int(np.argmax(probabilities))

    if class_index >= len(CLASS_NAMES):
        raise ValueError("CLASS_NAMES does not match the model output size.")

    return CLASS_NAMES[class_index], float(probabilities[class_index])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py path/to/plant_image.jpg")

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    disease, confidence = predict_image(image_path)
    print(f"Predicted disease: {disease}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
