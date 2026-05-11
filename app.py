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
# Load model (DenseNet201)
# -----------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path="densenet201_optimized.tflite"   # ✅ your model path
    )
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

.custom-subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 25px;
}

.stImage img {
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}

.prediction-card {
    background: linear-gradient(to right, #e0c3fc, #8ec5fc);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    color: #1e3a8a;
    margin-top: 20px;
}

.stInfo {
    background-color: #ede9fe !important;
    color: #7c3aed !important;
    border-radius: 10px;
    padding: 12px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="custom-header">🩺 Retinal Disease Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">Upload a retinal image to detect disease using AI</div>', unsafe_allow_html=True)

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Drag & drop a retinal image here or click to browse (JPG, PNG)",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Preprocessing
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

    st.image(image, caption="🖼️ Uploaded Retinal Image")

    st.info("🔍 Running model prediction...")

    input_data = preprocess(image)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred = np.argmax(output)
    confidence = output[pred]

    st.markdown(f"""
    <div class="prediction-card">
        ✅ Prediction: <b>{class_names[pred]}</b><br>
        📊 Confidence: {confidence*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👁️ Please upload a retinal image to begin diagnosis.")
