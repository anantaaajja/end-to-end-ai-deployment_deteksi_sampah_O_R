# train_model.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# === 1. Lokasi data ===
TRAIN_DIR = "DATASET/TRAIN"  # Sesuaikan jika path beda

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Folder TRAIN tidak ditemukan di: {TRAIN_DIR}")

# === 2. Generator untuk TRAIN (tanpa split, karena TEST sudah terpisah) ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Pastikan urutan kelas: O (organic) = 0, R (recyclable/anorganic) = 1
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    classes=['O', 'R']  # eksplisit: O → 0, R → 1
)

# === 3. Bangun model (Transfer Learning) ===
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # binary: 0=organic, 1=inorganic
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === 4. Latih model ===
print("🚀 Melatih model menggunakan folder TRAIN...")
history = model.fit(
    train_gen,
    epochs=15,  # Bisa dikurangi jadi 10 jika terlalu lama
    steps_per_epoch=len(train_gen)
)

# === 5. Simpan model & label ===
os.makedirs("model", exist_ok=True)
model.save("model/waste_classifier.h5")

# Mapping: index 0 → 'organic', index 1 → 'inorganic'
class_names = ["organic", "inorganic"]
with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

print("\n✅ Model berhasil disimpan di 'model/waste_classifier.h5'")
print("✅ Label: index 0 = organic, index 1 = inorganic")