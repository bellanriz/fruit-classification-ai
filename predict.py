import argparse
import os

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
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])
    predicted_class = class_names[predicted_index]
    return predicted_class, confidence, predictions[0]


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
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Run train.py first.")

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
    print("\nClass probabilities:")
    for idx, class_name in enumerate(class_names):
        print(f"  {class_name}: {prediction_vector[idx]:.4f}")


if __name__ == "__main__":
    main()
