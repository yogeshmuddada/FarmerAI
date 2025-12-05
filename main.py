import streamlit as st
from PIL import Image, UnidentifiedImageError
import io

# ---- Upload UI ----
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Convert uploaded file → PIL image
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data))

        st.image(image, caption="Uploaded Image", use_column_width=True)

    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid JPG/PNG.")
    else:
        # -------- YOUR ORIGINAL CODE (UNCHANGED) --------
        from PIL import Image, UnidentifiedImageError
        from transformers import ViTFeatureExtractor, ViTForImageClassification

        feature_extractor = ViTFeatureExtractor.from_pretrained(
            'wambugu71/crop_leaf_diseases_vit'
        )

        model = ViTForImageClassification.from_pretrained(
            'wambugu1738/crop_leaf_diseases_vit',
            ignore_mismatched_sizes=True
        )

        # HERE: pass uploaded image exactly to your code
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        st.success(f"Predicted class: {model.config.id2label[predicted_class_idx]}")
