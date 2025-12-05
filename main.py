# app.py
import os
import io
import streamlit as st
from PIL import Image, UnidentifiedImageError


if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
   
    st.warning("GOOGLE_API_KEY not found in .streamlit/secrets.toml. LLM calls will fail without it.")

# --- Page UI ---
st.set_page_config(page_title="Plant Disease Predictor — Upload Model & Image", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.write("Upload your image (Corn, Potato, Rice, Wheat). The app will predict the disease and provide a short explanation.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


@st.cache_resource
def load_model_and_extractor():
    
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    feature_extractor = ViTFeatureExtractor.from_pretrained('wambugu71/crop_leaf_diseases_vit')
    model = ViTForImageClassification.from_pretrained(
        'wambugu1738/crop_leaf_diseases_vit',
        ignore_mismatched_sizes=True
    )
    model.eval()
    return feature_extractor, model


def run_inference(image: Image.Image, feature_extractor, model):
    # ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    return predicted_class, logits

# --- Main flow ---
if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(bytes_data))
    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid JPG/PNG.")
    else:
        st.image(pil_image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        
        with st.spinner("Loading model (this may take some time for large models)..."):
            try:
                feature_extractor, model = load_model_and_extractor()
            except Exception as e:
                st.error(f"Error loading model or extractor: {e}")
                st.stop()

        
        with st.spinner("Running inference..."):
            try:
                predicted_class, logits = run_inference(pil_image, feature_extractor, model)
            except Exception as e:
                st.error(f"Error during model inference: {e}")
                st.stop()

        st.success(f"Predicted class: **{predicted_class}**")


        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as e:
            st.error(f"LLM library import failed: {e}\nMake sure `langchain_google_genai` is installed in your environment.")
            st.stop()

        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("Google API key not found in environment. Put it in .streamlit/secrets.toml as GOOGLE_API_KEY = \"...\"")
        else:
            with st.spinner("Generating explanation using AI..."):
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        
                  prompt = f"""
You are an agricultural expert. The detected plant condition is: '{predicted_class}'.

Your task:
1. If the class indicates a DISEASE, provide:
   - A simple explanation in very easy layman language that farmers can understand (max 100 words)
   - Precautions farmers should take for that crop and disease
   - Recommended pesticides (common, widely available options)
   - Organic or natural methods to control or cure the disease
   - Tips to improve yield

2. If the class is 'Healthy', provide:
   - Simple confirmation that the plant is healthy
   - General care tips to maintain good yield

Supported classes:
Corn: Common Rust, Gray Leaf Spot, Leaf Blight, Healthy
Potato: Early Blight, Late Blight, Healthy
Rice: Brown Spot, Hispa, Leaf Blast, Healthy
Wheat: Brown Rust, Yellow Rust, Healthy

Your answer must be short, clear, structured, and farmer-friendly.
"""

                    result = llm.invoke(prompt)

                    ai_response = getattr(result, "content", None) or str(result)

                except Exception as e:
                    ai_response = f"LLM call failed: {e}"

            st.subheader("AI Explanation")
            st.write(ai_response)

else:
    pass
