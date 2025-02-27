import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Ensure "uploads" directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load embeddings and filenames safely
try:
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('fileNames.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Missing 'embeddings.pkl' or 'filenames.pkl'. Ensure they exist before running.")
    st.stop()

# Load the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

st.title('Snap2Shop')

# Function to save uploaded file
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

# Upload file section
uploaded_file = st.file_uploader("📤 Upload an image for recommendation:")
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)

    if file_path:
        # Display uploaded image
        display_image = Image.open(file_path)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        # Feature extraction
        features = feature_extraction(file_path, model)

        # Get recommendations
        indices = recommend(features, feature_list)

        # Display recommended images
        st.subheader("✨ Recommended Products:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][i]], use_column_width=True)
    else:
        st.error("❌ Some error occurred in file upload.")
