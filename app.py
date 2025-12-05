# app.py
import streamlit as st
st.set_option("server.maxUploadSize", 300)  # allow up to 300 MB uploads

from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import tempfile
import sys

# ---------- Config ----------
IMG_TARGET_SIZE = (224, 224)  # change if your model expects another size
TOP_K = 5

# Replace with your actual class labels (order must match model training)
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

# ---------- UI ----------
st.set_page_config(page_title="Plant Disease Predictor — Upload Model & Image", layout="centered")
st.title("🌿 Plant Disease Classifier — Upload Model & Image")
st.write("Upload your Keras `.h5` model file (up to 300 MB) and an image. The app will load the model and predict top classes.")

uploaded_model = st.file_uploader("Upload Keras model file (.h5)", type=["h5"], key="model_uploader")
uploaded_image = st.file_uploader("Upload an image (jpg / png)", type=["jpg", "jpeg", "png"], key="image_uploader")

# ---------- Helpers ----------
@st.cache_resource
def load_model_from_bytes(model_bytes: bytes):
    """
    Save uploaded bytes to a temp file and load the Keras model from disk.
    compile=False is used to speed up loading when training config is not needed.
    """
    # Create a stable temporary file path (keras expects a filename)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    try:
        tmp_file.write(model_bytes)
        tmp_file.flush()
        tmp_file.close()
        # load_model may raise if model incompatible; compile=False often helps
        model = load_model(tmp_file.name, compile=False)
    finally:
        # we keep the file while model is loaded; remove file after load
        try:
            os.remove(tmp_file.name)
        except Exception:
            pass
    return model

def preprocess_pil(img: Image.Image, target_size=IMG_TARGET_SIZE):
    """Convert PIL image to model input array (batch of 1)."""
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_topk(model, img_array, top_k=TOP_K):
    preds = model.predict(img_array).flatten()
    top_idx = np.argsort(preds)[::-1][:top_k]
    result = [(class_labels[i] if i < len(class_labels) else f"Class_{i}", float(preds[i])) for i in top_idx]
    return result, preds

def sizeof_fmt(num, suffix='B'):
    # human readable file size
    for unit in ['','K','M','G','T','P']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

# ---------- App logic ----------
if uploaded_model is None:
    st.info("Please upload a `.h5` model file (max 300 MB).")
if uploaded_image is None:
    st.info("Please upload an image to classify.")

if uploaded_model is not None and uploaded_image is not None:
    try:
        # show model file size and name
        try:
            uploaded_model.seek(0, io.SEEK_END)
            model_size = uploaded_model.tell()
            uploaded_model.seek(0)
        except Exception:
            # fallback if file-like object doesn't support seek
            model_bytes_peek = uploaded_model.read()
            model_size = len(model_bytes_peek)
            uploaded_model.seek(0)
        st.write(f"**Model file:** {getattr(uploaded_model, 'name', 'uploaded_model.h5')}  —  {sizeof_fmt(model_size)}")
        if model_size > 300 * 1024 * 1024:
            st.warning("Uploaded model size exceeds 300 MB. Streamlit won't accept files larger than the configured limit.")
        # load model
        with st.spinner("Loading model (this may take some time for large models)..."):
            model_bytes = uploaded_model.read()
            model = load_model_from_bytes(model_bytes)
        st.success("Model loaded successfully.")

        # Read and display image
        image_bytes = uploaded_image.read()
        pil_img = Image.open(io.BytesIO(image_bytes))
        st.image(pil_img, caption="Input image", use_column_width=True)

        # Preprocess and predict
        x = preprocess_pil(pil_img)
        with st.spinner("Running prediction..."):
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
        # show helpful diagnostics
        st.error("Error while loading model or predicting.")
        st.exception(e)
        st.write("Tips / Debugging:")
        st.write("- Ensure the `.h5` model was trained with the same class order as `class_labels`.")
        st.write("- If your model used a different image preprocessing (mean subtraction / different size), update `preprocess_pil`.")
        st.write("- If model loading fails due to custom layers, you may need to supply custom_objects when calling `load_model`.")
        st.write("- For very large models consider converting to TensorFlow SavedModel format or using a smaller/quantized model.")
else:
    st.write("")  # keep layout tidy

st.markdown("---")
st.write("Notes:")
st.write("- The class label list must exactly match the order used during training.")
st.write("- If your model used different preprocessing (e.g., mean subtraction, other size), change `preprocess_pil` accordingly.")
st.write("- Large models need more RAM; Streamlit Cloud has resource limits. Consider model optimization if you run into memory errors.")
