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
import requests
from io import BytesIO

client_id = st.secrets["google_oauth"]["client_id"]
client_secret = st.secrets["google_oauth"]["client_secret"]
project_id = st.secrets["google_oauth"]["project_id"]
auth_uri = st.secrets["google_oauth"]["auth_uri"]
token_uri = st.secrets["google_oauth"]["token_uri"]
redirect_uris = st.secrets["google_oauth"]["redirect_uris"]


# Set page title and favicon
st.set_page_config(page_title="Snap2Shop", page_icon="🛍️")

# Create a temporary directory for uploads
UPLOAD_DIR = tempfile.mkdtemp()


# Function to load embeddings with error handling and caching
@st.cache_data
def load_embeddings():
    try:
        feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
        filenames = pickle.load(open('drive_filenames.pkl', 'rb'))
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


# Function to get image from Google Drive with caching
def get_optimized_google_drive_image(filename, width=300):
    try:
        # Check if the filename is already a Google Drive URL
        if filename.startswith('https://drive.google.com'):
            image_url = filename
        else:
            st.warning(f"The filename {filename} is not a Google Drive URL")
            return None

        # Google Drive requires special handling for direct downloads
        image_url = image_url.replace('export=view', 'export=download')

        # Fetch the image
        response = requests.get(image_url, timeout=10)

        if response.status_code == 200:
            try:
                # Open the image from bytes
                img = Image.open(BytesIO(response.content))

                # Optimize the image quality
                # 1. Resize while maintaining aspect ratio
                original_width, original_height = img.size
                aspect_ratio = original_height / original_width
                new_height = int(width * aspect_ratio)
                img = img.resize((width, new_height), Image.LANCZOS)  # LANCZOS is high-quality downsampling

                # 2. Optimize image quality
                output_buffer = BytesIO()

                # Determine format - convert to WebP if possible for better compression/quality ratio
                format_to_save = 'WEBP' if 'WEBP' in Image.registered_extensions().values() else img.format

                # Save with optimized quality
                if format_to_save == 'WEBP':
                    img.save(output_buffer, format=format_to_save, quality=85, method=6)
                elif format_to_save in ('JPEG', 'JPG'):
                    img.save(output_buffer, format=format_to_save, quality=85, optimize=True)
                else:
                    img.save(output_buffer, format=format_to_save, optimize=True)

                # Get the optimized image
                output_buffer.seek(0)
                optimized_img = Image.open(output_buffer)

                return optimized_img
            except Exception as e:
                st.error(f"Error processing image {filename}: {e}")
                return None
        else:
            st.warning(f"Could not fetch image {filename} (Status: {response.status_code})")
            return None
    except Exception as e:
        st.error(f"Error loading image {filename}: {e}")
        return None


# Add debugging tools
def add_debugging_section():
    with st.expander("🛠️ Debugging Tools"):
        # Test specific image retrieval
        test_index = st.number_input("Test image index (0 to length-1):", 0, len(filenames) - 1)

        if st.button("Test Image Retrieval"):
            if 0 <= test_index < len(filenames):
                filename = filenames[test_index]
                st.write(f"Testing filename: {filename}")

                img = get_optimized_google_drive_image(filename, width=300)

                if img:
                    st.success(f"✅ Successfully retrieved image")
                    st.image(img, caption=f"Retrieved image", width=200)
                else:
                    st.error("❌ Failed to retrieve image")


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

                # When displaying recommended images
                for i, col in enumerate(cols):
                    if i < len(indices[0]):
                        progress_bar.progress((i + 1) / 5)
                        with col:
                            with st.spinner(f"Loading image {i + 1}..."):
                                img = get_optimized_google_drive_image(filenames[indices[0][i]], width=300)
                                if img:
                                    st.image(img, use_column_width=True)
                                    st.markdown(f"**Product {i + 1}**")
                                else:
                                    st.error(f"Could not load image {i + 1}")
                progress_bar.empty()
            else:
                st.error("❌ Some error occurred in file upload.")

# Add debugging section at the bottom
add_debugging_section()

# Footer
st.markdown("---")
st.markdown("Snap2Shop - Find products you love with just a snap!")
