import argparse
import os

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH = "models/fruit_classifier.h5"
CLASS_INDICES_PATH = "models/class_indices.npy"


def load_class_names(class_indices_path: str) -> list[str]:
    class_indices = np.load(class_indices_path, allow_pickle=True).item()
    index_to_class = {idx: name for name, idx in class_indices.items()}
    return [index_to_class[i] for i in range(len(index_to_class))]


def predict_with_tta(model, image_path: str, img_size: tuple[int, int]) -> np.ndarray:
    # Average several transformed views to improve robustness on real-world photos.
    base = Image.open(image_path).convert("RGB")
    variants: list[Image.Image] = [
        base,
        ImageOps.mirror(base),
        base.rotate(-10, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)),
        base.rotate(10, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)),
        ImageEnhance.Brightness(base).enhance(0.9),
        ImageEnhance.Brightness(base).enhance(1.1),
        ImageEnhance.Contrast(base).enhance(1.1),
    ]

    batch: list[np.ndarray] = []
    for variant in variants:
        resized = variant.resize(img_size, Image.Resampling.BILINEAR)
        batch.append(image.img_to_array(resized) / 255.0)

    stacked = np.stack(batch, axis=0)
    predictions = model.predict(stacked, verbose=0)
    return np.mean(predictions, axis=0)


def predict_single_image(image_path: str, img_size: tuple[int, int]) -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )

    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(
            f"Class indices not found at {CLASS_INDICES_PATH}. Run train.py first."
        )

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_INDICES_PATH)

    probabilities = predict_with_tta(model, image_path, img_size)
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nClass probabilities:")
    for idx, class_name in enumerate(class_names):
        print(f"- {class_name}: {probabilities[idx] * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fruit class for a single image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=(100, 100),
        metavar=("WIDTH", "HEIGHT"),
        help="Image size expected by the model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_single_image(image_path=args.image, img_size=tuple(args.img_size))
