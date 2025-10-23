# app.py
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import json

# Load model & class names
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/waste_classifier.h5")
    with open("model/class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)[0][0]
    if prediction < 0.5:
        class_idx = 0  # organic
        confidence = 1.0 - prediction
    else:
        class_idx = 1  # inorganic
        confidence = prediction
    return class_names[class_idx], float(confidence)

# === Web UI ===
st.set_page_config(page_title="Waste Classifier", page_icon="♻️")
st.title("♻️ Waste Classifier (Organic vs Anorganic)")
st.write("Upload gambar sampah untuk prediksi!")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)
    
    with st.spinner("Memproses..."):
        class_name, confidence = predict(image)
    
    st.success(f"**Hasil Prediksi:** {class_name}")
    st.info(f"**Confidence:** {confidence:.2%}")