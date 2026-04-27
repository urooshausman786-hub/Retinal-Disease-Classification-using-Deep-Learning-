import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/model.keras")

model = load_model()

class_names = ["Disease1", "Disease2", "Disease3", "Disease4"]

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("Retinal Disease Classification")

uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess_image(image)
    prediction = model.predict(img)

    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Prediction: **{pred_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")
