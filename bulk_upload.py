import os
import cloudinary.uploader
import cloudinary.api
from config import cloudinary  # Import Cloudinary config

# Set the local folder where images are stored
IMAGE_FOLDER = "images"  # Change this to your image folder path

# Function to upload images
def upload_images():
    files = os.listdir(IMAGE_FOLDER)
    total_files = len(files)
    uploaded_count = 0

    for filename in files:
        file_path = os.path.join(IMAGE_FOLDER, filename)

        # Only upload files (ignore directories)
        if os.path.isfile(file_path):
            try:
                cloudinary.uploader.upload(file_path, folder="my_google_drive_uploads")
                uploaded_count += 1
                print(f"[{uploaded_count}/{total_files}] Uploaded: {filename}")
            except Exception as e:
                print(f"❌ Error uploading {filename}: {e}")

upload_images()
