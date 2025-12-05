# app.py
import os
import io
import streamlit as st
from PIL import Image, UnidentifiedImageError

# --- Load secret into env (Streamlit makes secrets available via st.secrets) ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # Not fatal here, but warn user in UI
    st.warning("GOOGLE_API_KEY not found in .streamlit/secrets.toml. LLM calls will fail without it.")

# --- Page UI ---
st.set_page_config(page_title="Plant Disease Predictor — Upload Model & Image", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.write("Upload your image (Corn, Potato, Rice, Wheat). The app will predict the disease and provide a short explanation.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# --- Cache model & feature extractor so they are loaded once per session/process ---
@st.cache_resource
def load_model_and_extractor():
    # keep your exact model-loading logic; change only to cache for performance
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    feature_extractor = ViTFeatureExtractor.from_pretrained('wambugu71/crop_leaf_diseases_vit')
    model = ViTForImageClassification.from_pretrained(
        'wambugu1738/crop_leaf_diseases_vit',
        ignore_mismatched_sizes=True
    )
    model.eval()
    return feature_extractor, model

# --- Helper: run inference (keeps your original inference calls unchanged) ---
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

        # Load model & extractor with spinner
        with st.spinner("Loading model (this may take some time for large models)..."):
            try:
                feature_extractor, model = load_model_and_extractor()
            except Exception as e:
                st.error(f"Error loading model or extractor: {e}")
                st.stop()

        # Run inference with spinner
        with st.spinner("Running inference..."):
            try:
                predicted_class, logits = run_inference(pil_image, feature_extractor, model)
            except Exception as e:
                st.error(f"Error during model inference: {e}")
                st.stop()

        st.success(f"Predicted class: **{predicted_class}**")

        # Generate LLM explanation (Gemini via LangChain's ChatGoogleGenerativeAI)
        # Wrap in spinner and try/except; require GOOGLE_API_KEY in env or st.secrets
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

                    # Clean, clear prompt ensuring predicted_class is interpolated
                    prompt = (
                        f"Explain the plant disease '{predicted_class}' in simple layman language for farmers. "
                        "Make the explanation very practical and easy to follow. "
                        "Limit the explanation to 100 words."
                    )

                    result = llm.invoke(prompt)
                    # `result.content` per prior usage — adjust if your SDK returns differently
                    ai_response = getattr(result, "content", None) or str(result)

                except Exception as e:
                    ai_response = f"LLM call failed: {e}"

            st.subheader("AI Explanation")
            st.write(ai_response)

else:
    # If no file uploaded, we show nothing extra (per your earlier request)
    pass
