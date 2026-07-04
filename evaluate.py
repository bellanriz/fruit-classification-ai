import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


MODEL_PATH = "models/fruit_classifier.h5"
CLASS_INDICES_PATH = "models/class_indices.npy"
TEST_DIR = "dataset/test"
RESULTS_DIR = "results"


def load_class_names(class_indices_path: str) -> list[str]:
    class_indices = np.load(class_indices_path, allow_pickle=True).item()
    index_to_class = {idx: name for name, idx in class_indices.items()}
    return [index_to_class[i] for i in range(len(index_to_class))]


def evaluate(batch_size: int, img_size: tuple[int, int]) -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )

    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(
            f"Class indices not found at {CLASS_INDICES_PATH}. Run train.py first."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_INDICES_PATH)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
    print("\nClassification Report:\n")
    print(report)

    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    plt.savefig(cm_path)
    plt.show()

    print(f"\nSaved report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained fruit classifier.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation data loader.",
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
    evaluate(batch_size=args.batch_size, img_size=tuple(args.img_size))
