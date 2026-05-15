
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("diabetic_retinopathy_model.h5")

# Class labels
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# App title
st.title("🩺 Diabetic Retinopathy Detection")

st.write("Upload a retinal image and let the AI analyze it.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    st.image(image, caption="Uploaded Retinal Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))

    image_array = np.array(image) / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    # Results
    st.subheader(f"Prediction: {predicted_class}")

    st.write(f"Confidence: {confidence:.2f}%")
