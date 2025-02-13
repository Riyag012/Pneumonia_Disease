import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource  # Cache model to avoid reloading
def load_model():
    return tf.keras.models.load_model("model_vgg16.keras")

model = load_model()

# Define class labels
CLASS_LABELS = ["normal", "opacity"]

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert("RGB")  # Convert to 3-channel (RGB)
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_pneumonia(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)[0]  # Get prediction array
    predicted_label = CLASS_LABELS[int(prediction > 0.5)]  # Map to label
    confidence = float(prediction[0])  # Get confidence score
    return predicted_label, confidence

# Streamlit UI
st.title("🩺 Pneumonia Detection from Chest X-ray")

st.write("""
This AI model predicts whether a chest X-ray is **Normal** or shows **Pneumonia**.
Upload an X-ray image to get the result.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a Chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    st.write("🔄 **Classifying...**")
    predicted_label, confidence = predict_pneumonia(img)

     # Map "opacity" to "Pneumonia" for better readability
    display_label = "Pneumonia" if predicted_label == "opacity" else "Normal"

    # Display result
    if display_label == "Pneumonia":
        st.error(f"🚨 **Prediction: Pneumonia (Confidence: {confidence:.2%})**")
    else:
        st.success(f"✅ **Prediction: Normal (Confidence: {confidence:.2%})**")