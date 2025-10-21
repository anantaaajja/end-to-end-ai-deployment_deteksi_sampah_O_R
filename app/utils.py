# app/utils.py
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Path model & label
MODEL_PATH = "model/waste_classifier.h5"
CLASS_NAMES_PATH = "model/class_names.json"

# Validasi keberadaan file
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model tidak ditemukan! Jalankan 'train_model.py' terlebih dahulu.")
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError("File class_names.json tidak ditemukan!")

# Load model dan label
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)  # Harus: ["organic", "inorganic"]

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess gambar untuk prediksi"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    return img_array

def predict_image(image_path):
    """
    Prediksi gambar sampah.
    Mengembalikan: (class_name, confidence)
    """
    img = preprocess_image(image_path)
    prediction = model.predict(img, verbose=0)[0][0]  # Nilai antara 0.0 - 1.0

    # Karena:
    # - Jika gambar = organic (O) → label = 0 → model output mendekati 0
    # - Jika gambar = inorganic (R) → label = 1 → model output mendekati 1
    #
    # Maka:
    if prediction < 0.5:
        class_idx = 0  # organic
        confidence = 1.0 - prediction
    else:
        class_idx = 1  # inorganic
        confidence = prediction

    class_name = class_names[class_idx]
    return class_name, float(confidence)