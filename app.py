import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("leaf_disease_model.h5")

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

def predict(img):
    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌿 Leaf Disease Detector",
    description="Upload a leaf image to detect disease"
)

interface.launch(share=True)