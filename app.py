import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="PM10 Prediction", layout="centered")
st.title("🌤️ PM10 Air Quality Prediction with Blur Detection")

# -----------------------------
# Blurriness Detection Function
# -----------------------------
def is_blurry(pil_img, threshold=100):
    img = np.array(pil_img.convert("L"))  # convert to grayscale
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return lap_var < threshold

# -----------------------------
# PM10 Advisory Generator
# -----------------------------
def pm10_advisory(pm10):
    if pm10 <= 50:
        return "Good air quality. You can go outside freely. No precautions needed."
    elif pm10 <= 100:
        return "Moderate air quality. People with breathing issues should limit outdoor activity."
    elif pm10 <= 250:
        return "Poor air quality. Avoid outdoor exercise. Use a mask if stepping out."
    else:
        return "Hazardous air quality. Stay indoors. Use air purifiers and wear masks if outside."

# -----------------------------
# Model Loader
# -----------------------------
@st.cache_resource
def load_model(weight_path="vgg16_aqi.best.hdf5"):
    base_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    for layer in base_model.layers:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    model.load_weights(weight_path)
    return model

model = load_model()

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a sky image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    if is_blurry(image_pil):
        st.warning("\u26a0\ufe0f The uploaded image seems blurry. Please retake a clear photo of the sky.")
    else:
        img = image_pil.resize((224, 224))
        img = np.array(img)
        img = tf.keras.applications.vgg16.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        pm10_value = float(prediction[0][0])

        st.markdown(f"### 🌫️ Predicted PM10: `{pm10_value:.2f} µg/m³`")
        st.markdown(f"### 🛡️ Advisory: {pm10_advisory(pm10_value)}")
