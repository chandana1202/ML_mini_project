import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bird Species Identification",
    page_icon="🐦",
    layout="centered"
)

IMG_SIZE = 224

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bird_species_model.keras"
)

CLASSES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bird_species_classes.json"
)

# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )

# --------------------------------------------------
# 3. Load bird species names
# --------------------------------------------------

@st.cache_data
def load_classes():
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            f"Class names file not found: {CLASSES_PATH}"
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    # Support either a list of names or a dictionary
    # containing class IDs and species names.
    if isinstance(names, list):
        return names

    if isinstance(names, dict):
        try:
            return [
                names[key]
                for key in sorted(names, key=lambda x: int(x))
            ]
        except (ValueError, TypeError):
            return list(names.values())

    raise ValueError(
        "The class names JSON must contain a list or dictionary."
    )

# --------------------------------------------------
# 4. Display application
# --------------------------------------------------

st.title("🐦 Bird Species Identification")
st.write(
    "Upload a bird photograph and let the trained "
    "MobileNetV2 model predict its species."
)

st.info(
    "This model was trained on the CUB-200-2011 dataset "
    "and recognizes 200 bird species."
)

# Load files and display helpful errors if something fails.
try:
    model = load_model()
    class_names = load_classes()

except Exception as error:
    st.error("The application could not load the model or class names.")
    st.write("Check that both files are in the same folder as app.py.")
    st.code(str(error))
    st.stop()

st.success("Model loaded successfully!")

# --------------------------------------------------
# 5. Upload an image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a bird image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(
            image,
            caption="Uploaded bird image",
            use_container_width=True
        )

        if st.button("Identify Bird", type="primary"):
            with st.spinner("Analyzing the image..."):

                # Resize the image to the model input size.
                resized_image = image.resize(
                    (IMG_SIZE, IMG_SIZE)
                )

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                image_batch = np.expand_dims(
                    image_array,
                    axis=0
                )

                # The model includes preprocessing internally.
                predictions = model.predict(
                    image_batch,
                    verbose=0
                )[0]

                if len(predictions) != len(class_names):
                    st.error(
                        "The model output count does not match "
                        "the number of species names in the JSON file."
                    )
                    st.stop()

                # Find the highest-scoring predictions.
                top_indices = np.argsort(predictions)[::-1][:5]
                best_index = int(top_indices[0])
                best_species = class_names[best_index]
                confidence = float(predictions[best_index]) * 100

            st.subheader("Prediction")
            st.success(f"Predicted species: {best_species}")

            st.metric(
                label="Model confidence",
                value=f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee that the identification is correct."
            )

            st.subheader("Top 5 predictions")

            for index in top_indices:
                species = class_names[int(index)]
                score = float(predictions[index]) * 100

                st.write(f"**{species}** — {score:.2f}%")
                st.progress(
                    min(max(float(predictions[index]), 0.0), 1.0)
                )

    except Exception as error:
        st.error("Unable to process this image.")
        st.code(str(error))

# --------------------------------------------------
# 6. Footer
# --------------------------------------------------

st.divider()
st.caption(
    "Machine Learning Mini-Project | "
    "Bird Species Identification from an Image"
)