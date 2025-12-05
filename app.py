# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

st.set_page_config(page_title="Plant Disease Predictor — Upload Model & Image", layout="centered")

IMG_TARGET_SIZE = (224, 224)
TOP_K = 5

# Replace with your actual class labels (keep order same as model training)
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

st.title("🌿 Plant Disease Classifier — Upload Model & Image")
st.write("Upload your Keras `.h5` model file *and* an image. The app will load the uploaded model (cached) and predict the image.")

# -------- Upload widgets --------
uploaded_model = st.file_uploader("Upload Keras model file (.h5)", type=["h5"], key="model_uploader")
uploaded_image = st.file_uploader("Upload an image (jpg / png)", type=["jpg", "jpeg", "png"], key="image_uploader")

# Cache model load using uploaded bytes so it doesn't reload repeatedly
@st.cache_resource
def load_model_from_bytes(model_bytes: bytes):
    tmp_path = "/tmp/uploaded_model.h5"
    with open(tmp_path, "wb") as f:
        f.write(model_bytes)
    model = load_model(tmp_path)
    return model

def preprocess_pil(img: Image.Image, target_size=IMG_TARGET_SIZE):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_topk(model, img_array, top_k=TOP_K):
    preds = model.predict(img_array).flatten()
    top_idx = np.argsort(preds)[::-1][:top_k]
    return [(class_labels[i] if i < len(class_labels) else f"Class_{i}", float(preds[i])) for i in top_idx], preds

# -------- Run prediction when both files are provided --------
if uploaded_model is None:
    st.info("Please upload a `.h5` model file.")
if uploaded_image is None:
    st.info("Please upload an image to classify.")

if uploaded_model is not None and uploaded_image is not None:
    try:
        with st.spinner("Loading model..."):
            # read bytes and load (cached)
            model_bytes = uploaded_model.read()
            model = load_model_from_bytes(model_bytes)

        st.success("Model loaded (from upload).")

        # open image
        image_bytes = uploaded_image.read()
        pil_img = Image.open(io.BytesIO(image_bytes))
        st.image(pil_img, caption="Input image", use_column_width=True)

        # preprocess & predict
        x = preprocess_pil(pil_img)
        topk, full_preds = predict_topk(model, x, TOP_K)

        st.subheader("Top predictions")
        for label, prob in topk:
            st.write(f"**{label}** — {prob:.4f}")

        if st.checkbox("Show full probabilities"):
            import pandas as pd
            df = pd.DataFrame({
                "class_index": list(range(len(full_preds))),
                "class_label": [class_labels[i] if i < len(class_labels) else f"Class_{i}" for i in range(len(full_preds))],
                "probability": full_preds
            }).sort_values("probability", ascending=False).reset_index(drop=True)
            st.dataframe(df)

    except Exception as e:
        st.error(f"Error while loading model or predicting: {e}")

st.markdown("---")
st.write("Notes:")
st.write("- The class label list must exactly match the order used during model training.")
st.write("- If your model used a different preprocessing (e.g., mean subtraction, different size), change `preprocess_pil` accordingly.")
