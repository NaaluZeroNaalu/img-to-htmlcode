import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
from PIL import Image  # Import Image module from PIL library
import cv2
from skimage.metrics import structural_similarity as ssim
import tempfile

# Load the saved model
model = tf.keras.models.load_model('C:\\Users\\ajith\\Desktop\\saranya\\img2html.h5')

# Define the class names
class_names = ['dropdownbar', 'footer', 'gradientcolorloginpage', 'loginpage', 'normalpages', 'searchpage', 'sidebar', 'topnavbar']

uploaded_file = st.file_uploader('Upload an Image')

if uploaded_file is not None:
    st.image(uploaded_file, width=200)

    # Convert the uploaded file to a PIL image
    pil_image = Image.open(uploaded_file)
    pil_image = pil_image.resize((200, 200))  # Resize the image to match the model's input size

    # Convert the PIL image to a NumPy array
    img_array = image.img_to_array(pil_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the pixel values

    # Predict the class probabilities
    predictions = model.predict(img_array)

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions[0])  # Get the index of the class with highest probability
    predicted_class = class_names[predicted_class_index]  # Get the predicted class name

    # Convert the predicted class name to a string
    predicted_class_str = str(predicted_class)

    # Print the predicted class
    st.write("Predicted class:", predicted_class_str)
    current_directory = os.getcwd()
    current_directory_name = os.path.basename(current_directory)
    # Load the CSV file containing image paths and codes
    data = pd.read_csv(predicted_class_str + ".csv", encoding='latin1')

    # Save uploaded image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.close()
    uploaded_file.seek(0)
    with open(temp_file.name, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Calculate similarity with each image in the dataset
    max_similarity = -1
    similar_code = None
    input_image = cv2.imread(temp_file.name)
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    for image_path, code in zip(data['picture'], data['code']):
        full_image_path = current_directory+image_path
        # Load the image
        print("Image path:", full_image_path)
        image = cv2.imread(full_image_path)
        print("Image:", image)
        
        # Check if the image is loaded successfully
        if image is None:
            print("Error: Unable to load image:", full_image_path)
            continue
    
        # Resize the image to match the size of the input_image
        image_resized = cv2.resize(image, (input_image.shape[1], input_image.shape[0]))
    
        # Convert the resized image to RGB color space
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
        # Calculate similarity
        similarity = ssim(input_image_gray, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY))
        
        if similarity > max_similarity:
            max_similarity = similarity
            similar_code = code

    st.write("Most similar code:", similar_code)
