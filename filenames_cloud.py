from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow
from numpy.linalg import norm
import numpy as np
import os
from tqdm import tqdm
import pickle

# Load the model (same as before)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


# Feature extraction function (same as before)
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


# Original filenames list
original_filenames = []
for file in os.listdir('images'):
    original_filenames.append(os.path.join('images', file))

# Create new filenames list for Cloudinary
cloudinary_filenames = []
feature_list = []

# Process each file
for file in tqdm(original_filenames):
    # Extract just the filename without the 'images/' directory prefix
    filename = os.path.basename(file)

    # Add the filename to cloudinary_filenames list
    # Use the filename only, without the path
    cloudinary_filenames.append(filename)

    # Extract features (same as before)
    feature_list.append(extract_features(file, model))

# Save the new cloudinary_filenames list
# pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(cloudinary_filenames, open('filenames.pkl', 'wb'))

print(f"Done! Created new embeddings.pkl and fileNames.pkl with {len(cloudinary_filenames)} entries.")
print("Example of stored filename:", cloudinary_filenames[0])