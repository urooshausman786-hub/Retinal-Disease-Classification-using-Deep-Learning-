import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="🩺 Retinal Disease Classification",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------
# Load TFLite model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="densenet201_optimized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_names = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract"]

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #ffffff, #f3e8ff);
}
.custom-header {
    color: #7c3aed;
    text-align: center;
    font-size: 45px;
    font-weight: bold;
}
.prediction-card {
    background: linear-gradient(to right, #e0c3fc, #8ec5fc);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="custom-header">🩺 Retinal Disease Classification</div>', unsafe_allow_html=True)

# -----------------------------

# File uploader
# -----------------------------
uploaded_file = st.file_uploader( "📤 Drag & drop a retinal image here or click to browse (JPG, PNG, max 200MB)", type=["jpg", "jpeg", "png"] )
# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    st.info("Running prediction...")

    input_data = preprocess(image)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred = np.argmax(output)
    confidence = output[pred]

    st.markdown(f"""
    <div class="prediction-card">
        Prediction: <b>{class_names[pred]}</b><br>
        Confidence: {confidence*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Please upload an image.")
