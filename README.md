# Fruit Classification AI

An image classification program that uses a Convolutional Neural Network (CNN) to recognize and classify 5 types of fruits: **Apple, Banana, Grape, Mango, and Orange**.

Built as a group project for the Image Classification using CNN assignment.

---

## Tech Stack

| Category | Tool / Library | Purpose |
|----------|---------------|---------|
| Language | Python 3.x | Main programming language |
| Deep Learning | TensorFlow / Keras | Build and train the CNN model |
| Data Handling | NumPy | Array and numerical operations |
| Image Processing | Pillow (PIL) | Load and resize images |
| Visualization | Matplotlib | Plot accuracy/loss graphs and confusion matrix |
| ML Metrics | Scikit-learn | Accuracy, precision, recall, F1-score, confusion matrix |
| GUI | Tkinter | Desktop interface for image upload and prediction |
| Dataset | Fruits 360 (Kaggle) | Pre-labelled fruit images at 100x100px |

---

## Project Flow

```
1. Dataset Collection
   └── Fruits 360 dataset from Kaggle
       ├── 5 classes: Apple, Banana, Grape, Mango, Orange
       ├── ~490 images per class for training
       └── ~160 images per class for testing

2. Data Preprocessing (train.py)
   ├── Resize images to 100x100px
   ├── Normalize pixel values (0-255 → 0-1)
   └── Apply data augmentation (rotation, flip, zoom)

3. CNN Model Architecture (train.py)
   ├── Conv2D + MaxPooling layers (feature extraction)
   ├── Dropout layers (prevent overfitting)
   ├── Flatten + Dense layers (classification)
   └── Output: Softmax (5 classes)

4. Model Training (train.py)
   ├── Optimizer: Adam
   ├── Loss: Categorical Crossentropy
   ├── Epochs: 20
   └── Saves trained model → models/fruit_classifier.h5

5. Model Evaluation (evaluate.py)
   ├── Accuracy, Precision, Recall, F1-score
   └── Confusion Matrix visualization → results/

6. Single Image Prediction (predict.py)
   └── Load model → preprocess image → predict class

7. GUI Application (gui.py)
   ├── Upload any fruit image
   └── Displays predicted fruit class + confidence score
```

---

## Folder Structure

```
fruit-classification-ai/
├── dataset/
│   ├── train/
│   │   ├── apple/
│   │   ├── banana/
│   │   ├── grape/
│   │   ├── mango/
│   │   └── orange/
│   └── test/
│       ├── apple/
│       ├── banana/
│       ├── grape/
│       ├── mango/
│       └── orange/
├── models/               # Saved trained model
├── results/              # Confusion matrix and plots
├── train.py              # CNN model building and training
├── evaluate.py           # Model evaluation and metrics
├── predict.py            # Single image prediction
├── gui.py                # Tkinter GUI application
└── requirements.txt      # Python dependencies
```

---

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model and save the artifacts
./env312/bin/python train.py

# Evaluate the saved model using the test data
./env312/bin/python evaluate.py

# Predict a single image
./env312/bin/python predict.py --image dataset/test/apple/3_100.jpg

# Launch the GUI application
./env312/bin/python gui.py
```

### Notes

- `train.py` saves the trained model to `models/fruit_classifier.h5` and class mappings to `models/class_indices.npy`.
- `evaluate.py` writes the classification report and confusion matrix to `results/`.
- `predict.py` prints the top prediction and class probability ranking.
- If the GUI fails to launch, make sure your Python environment includes `tkinter`.

---

## Dataset

- **Source:** [Fruits 360 Dataset on Kaggle](https://www.kaggle.com/datasets/moltean/fruits)
- **Classes:** Apple, Banana, Grape, Mango, Orange
- **Image size:** 100 x 100 pixels
- **Train images:** ~490 per class
- **Test images:** ~160 per class
