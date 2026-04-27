import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(
        model_path="model/densenet201_optimized.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ⚠️ Put your REAL classes here
class_names = ["Disease1", "Disease2", "Disease3", "Disease4"]

IMG_SIZE = (224, 224)

def preprocess(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

st.title("Retinal Disease Classification")

file = st.file_uploader("Upload Fundus Image", type=["jpg", "png"])

if file:
    img = Image.open(file)
    st.image(img, use_column_width=True)

    input_data = preprocess(img)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred = np.argmax(output)
    conf = np.max(output)

    st.success(f"Prediction: {class_names[pred]}")
    st.info(f"Confidence: {conf:.2f}")
