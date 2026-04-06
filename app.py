import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# -------------------------------
# Load Models (Update paths)
# -------------------------------
@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("cnn_model.h5")
    model2 = tf.keras.models.load_model("deepfake_model.h5")
    return model1, model2

model1, model2 = load_models()

# -------------------------------
# Helper: Preprocess Image
# -------------------------------
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# Prediction Function
# -------------------------------
def predict(model, image):
    processed = preprocess_image(image)
    pred = model.predict(processed)

    if pred[0][0] > 0.5:
        return "Fake"
    else:
        return "Real"

# -------------------------------
# UI
# -------------------------------
st.title("🧠 Deepfake Detection App")

# Model selection
model_choice = st.selectbox(
    "Select Model",
    ("cnn_model", "deepfake_model")
)

selected_model = model1 if model_choice == "Model 1" else model2

# -------------------------------
# Image Upload Section
# -------------------------------
st.header("📤 Upload Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Inference"):
        result = predict(selected_model, image)
        st.success(f"Prediction: {result}")

# -------------------------------
# Webcam Section
# -------------------------------
st.header("📷 Live Camera")

run_camera = st.toggle("Turn Camera ON/OFF")

if run_camera:
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict on frame
        processed = preprocess_image(rgb_frame)
        pred = selected_model.predict(processed)

        label = "Fake" if pred[0][0] > 0.5 else "Real"

        # Put text on frame
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()