import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
from io import BytesIO

# Set page title and favicon
st.set_page_config(page_title="Snap2Shop", page_icon="🛍️")

# Set up Cloudinary configuration
cloudinary.config(
    cloud_name=st.secrets["ramhere"],
    api_key=st.secrets["333688213841244"],
    api_secret=st.secrets["PsNKMyrVc4iB3AeswfxCDOWkspk"]
)

# Create a temporary directory for uploads
UPLOAD_DIR = tempfile.mkdtemp()


# Function to load embeddings with error handling and caching
@st.cache_data
def load_embeddings():
    try:
        feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
        filenames = pickle.load(open('fileNames.pkl', 'rb'))
        return feature_list, filenames
    except FileNotFoundError:
        st.error("❌ Missing 'embeddings.pkl' or 'filenames.pkl'. Ensure they exist before running.")
        st.stop()


# Load embeddings and filenames
feature_list, filenames = load_embeddings()


# Load the model with caching
@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    return tf.keras.Sequential([model, GlobalMaxPooling2D()])


model = load_model()

# App title and description
st.title('Snap2Shop 🛍️')
st.markdown("Upload a product image to find similar items!")


# Function to save uploaded file temporarily
def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"❌ Error saving file: {e}")
        return None


# Feature extraction function
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


# Recommendation function
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# Function to get image from Cloudinary with caching
# Function to get image from Cloudinary with caching
@st.cache_data
def get_cloudinary_image(filename):
    try:
        # The filename is now just the base filename without path
        # e.g., "product123.jpg" instead of "images/product123.jpg"

        # Remove file extension to get the public_id
        public_id = os.path.splitext(filename)[0]

        # If you uploaded to a specific folder in Cloudinary, include it here
        # For example, if you uploaded to a folder called "snap2shop":
        public_id = f"my_google_drive_uploads/{public_id}"

        # Create Cloudinary URL
        image_url = cloudinary.utils.cloudinary_url(
            public_id,  # Use the filename without extension
            secure=True,
            width=300,
            crop="scale",
            quality="auto:good",  # Auto-optimize quality
            fetch_format="auto"  # Use the best format for the browser
        )[0]

        # Fetch the image
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            st.warning(f"Could not fetch image {filename} (Status: {response.status_code})")
            return None
    except Exception as e:
        st.error(f"Error loading image {filename}: {e}")
        return None

# Main app flow
with st.container():
    # Upload file section
    uploaded_file = st.file_uploader("📤 Upload an image for recommendation:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner("Processing your image..."):
            file_path = save_uploaded_file(uploaded_file)

            if file_path:
                # Display uploaded image
                col1, col2 = st.columns([1, 2])
                with col1:
                    display_image = Image.open(file_path)
                    st.image(display_image, caption="Uploaded Image", use_column_width=True)

                with col2:
                    st.info("Finding similar products...")

                # Feature extraction
                features = feature_extraction(file_path, model)

                # Get recommendations
                indices = recommend(features, feature_list)

                # Display recommended images
                st.subheader("✨ Recommended Products:")
                cols = st.columns(5)

                # Progress indicator
                progress_bar = st.progress(0)

                for i, col in enumerate(cols):
                    if i < len(indices[0]):
                        progress_bar.progress((i + 1) / 5)
                        with col:
                            with st.spinner(f"Loading image {i + 1}..."):
                                img = get_cloudinary_image(filenames[indices[0][i]])
                                if img:
                                    st.image(img, use_column_width=True)
                                    st.markdown(f"**Product {i + 1}**")

                progress_bar.empty()
            else:
                st.error("❌ Some error occurred in file upload.")

# Footer
st.markdown("---")
st.markdown("Snap2Shop - Find products you love with just a snap!")
