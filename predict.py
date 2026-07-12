import argparse
import os
import sys

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "models/fruit_classifier.h5"
CLASS_INDICES_PATH = "models/class_indices.npy"
IMG_SIZE = (100, 100)


def load_class_names(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class indices file not found: {path}. Run train.py first.")

    class_indices = np.load(path, allow_pickle=True).item()
    class_names = [None] * len(class_indices)
    for name, index in class_indices.items():
        class_names[index] = name
    return class_names


def preprocess_image(image_path, target_size):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return image_array


def predict_image(model, class_names, image_path, img_size):
    image_array = preprocess_image(image_path, img_size)
    image_batch = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_batch, verbose=0)
    prediction_vector = predictions[0]
    predicted_index = int(np.argmax(prediction_vector))
    confidence = float(prediction_vector[predicted_index])
    predicted_class = class_names[predicted_index]
    return predicted_class, confidence, prediction_vector


def format_top_predictions(class_names, prediction_vector, top_k=3):
    ranked = sorted(
        enumerate(prediction_vector), key=lambda item: item[1], reverse=True
    )[:top_k]
    return [(class_names[index], float(score)) for index, score in ranked]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the fruit class for a single image.")
    parser.add_argument("--image", "-i", required=True, help="Path to the input image file.")
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=IMG_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help="Target image size for the model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}. Run train.py first.", file=sys.stderr)
        return 1

    if not os.path.exists(CLASS_INDICES_PATH):
        print(
            f"ERROR: Class indices file not found: {CLASS_INDICES_PATH}. Run train.py first.",
            file=sys.stderr,
        )
        return 1

    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        class_names = load_class_names(CLASS_INDICES_PATH)

        predicted_class, confidence, prediction_vector = predict_image(
            model, class_names, args.image, tuple(args.img_size)
        )

        print("\n=== Prediction ===")
        print(f"Image: {args.image}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")

        top_predictions = format_top_predictions(class_names, prediction_vector, args.top_k)
        print("\nTop predictions:")
        for rank, (class_name, score) in enumerate(top_predictions, start=1):
            print(f"  {rank}. {class_name} ({score:.4f})")

        print("\nAll class probabilities:")
        for idx, class_name in enumerate(class_names):
            print(f"  {class_name}: {prediction_vector[idx]:.4f}")

        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
