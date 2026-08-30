import os
import json
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "leaf_disease_model.h5"
)

# Class names path
CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)


# Check if required files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )


# Load model
model = load_model(MODEL_PATH)


# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


# Prediction function
def predict(img):
    if img is None:
        return "Please upload a leaf image."

    # Resize image
    img = img.resize((224, 224))

    # Convert image to NumPy array
    img_array = np.array(img)

    # Make sure image has 3 channels
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Normalize pixel values
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array, verbose=0)

    # Get predicted class
    predicted_index = np.argmax(prediction[0])

    predicted_class = class_names[predicted_index]

    return predicted_class


# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌿 Leaf Disease Detector",
    description="Upload a leaf image to detect disease"
)


# Launch application
interface.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)