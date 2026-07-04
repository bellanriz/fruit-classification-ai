import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageEnhance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_PATH = "models/fruit_classifier.h5"
RESULTS_DIR = "results"

# Hyperparameters
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 20
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def boost_target_classes(
    train_dir: str,
    target_classes: tuple[str, ...] = ("mango", "grape", "rambutan"),
    variants_per_image: int = 2,
) -> None:
    """Create extra enhanced images for difficult classes to improve recall."""
    for class_name in target_classes:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[boost] Skipped missing class folder: {class_dir}")
            continue

        all_files = [
            name
            for name in os.listdir(class_dir)
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        ]

        source_files = [name for name in all_files if not name.startswith("boost_")]
        created_count = 0

        for file_name in source_files:
            file_path = os.path.join(class_dir, file_name)
            try:
                base = Image.open(file_path).convert("RGB")
            except Exception:
                continue

            stem = os.path.splitext(file_name)[0]
            variants = [
                ImageEnhance.Color(base).enhance(1.15),
                ImageEnhance.Contrast(base).enhance(1.18),
                base.rotate(-8, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)),
                base.rotate(8, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)),
            ]

            for idx, variant in enumerate(variants[:variants_per_image], start=1):
                out_name = f"boost_{stem}_{idx}.jpg"
                out_path = os.path.join(class_dir, out_name)
                variant.save(out_path, format="JPEG", quality=95)
                created_count += 1

        print(f"[boost] {class_name}: created {created_count} enhanced images")


def validate_class_folders(train_dir: str, test_dir: str) -> None:
    train_classes = {
        name
        for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name))
    }
    test_classes = {
        name
        for name in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, name))
    }

    missing_in_test = sorted(train_classes - test_classes)
    missing_in_train = sorted(test_classes - train_classes)
    if missing_in_test or missing_in_train:
        details = []
        if missing_in_test:
            details.append(f"Missing in test: {', '.join(missing_in_test)}")
        if missing_in_train:
            details.append(f"Missing in train: {', '.join(missing_in_train)}")
        raise ValueError("Train/test class folders must match. " + " | ".join(details))


boost_target_classes(TRAIN_DIR)
validate_class_folders(TRAIN_DIR, TEST_DIR)

#Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Only normalize test images (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Save class indices for later use in predict.py and gui.py
class_indices = train_generator.class_indices
print("Class indices:", class_indices)
np.save("models/class_indices.npy", class_indices)
num_classes = len(class_indices)

# CNN MODEL
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])  

model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', 
  metrics=['accuracy'])

model.summary()

# Callbacks
checkpoint = ModelCheckpoint(MODEL_PATH,save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stopping]
)

# Plot accuracy and loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_results.png'))
plt.show()

print(f"Training completed. Model saved to {MODEL_PATH}. Training results saved to {RESULTS_DIR}/training_results.png.")
print(f"Training plot saved to {RESULTS_DIR}/training_results.png.")

