import streamlit as st
from PIL import Image, UnidentifiedImageError
import io
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os


os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ---- Page UI ----
st.set_page_config(page_title="Plant Disease Predictor", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.write("Upload your image (Corn, Potato, Rice, Wheat). The app will predict the disease and explain it in simple terms.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

predicted_class = None  # to store prediction globally

if uploaded_file is not None:
    try:
        # Convert uploaded file → PIL image
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data))

        st.image(image, caption="Uploaded Image", use_column_width=True)

    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid JPG/PNG.")
    else:
        # Import only when needed
        from transformers import ViTFeatureExtractor, ViTForImageClassification

        # Show spinner while loading the model
        with st.spinner("Loading model (this may take some time for large models)..."):
            feature_extractor = ViTFeatureExtractor.from_pretrained(
                "wambugu71/crop_leaf_diseases_vit"
            )

            model = ViTForImageClassification.from_pretrained(
                "wambugu1738/crop_leaf_diseases_vit",
                ignore_mismatched_sizes=True
            )

        # Show spinner while running inference
        with st.spinner("Running inference..."):
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = model.config.id2label[predicted_class_idx]

        st.success(f"Predicted class: **{predicted_class}**")

        # --- AI Explanation Section ---
        with st.spinner("Generating explanation using AI..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

            prompt = f"""
            Explain the plant disease **{predicted_class}** in simple layman language.
            Make it easy for farmers to understand.
            Limit the explanation to 100 words.
            """

            result = llm.invoke(prompt)
            ai_response = result.content

        st.subheader("AI Explanation")
        st.write(ai_response)
