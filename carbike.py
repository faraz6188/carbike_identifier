# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.utils import load_img, img_to_array

# # Title
# st.title("Car or Bike Identifier")

# # Load the model and class indices
# model = tf.keras.models.load_model("car_bike_classifier.h5")
# class_indices = np.load("class_indices.npy", allow_pickle=True).item()

# # File uploader
# uploaded_file = st.file_uploader("Upload an image of a Car or Bike", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#     # Preprocess the uploaded image
#     test_image = load_img(uploaded_file, target_size=(64, 64))
#     test_image_array = img_to_array(test_image) / 255.0
#     test_image_expanded = np.expand_dims(test_image_array, axis=0)

#     # Predict
#     result = model.predict(test_image_expanded)
#     if result[0][0] > 0.5:
#         prediction = f"It's a {list(class_indices.keys())[1]}"  # Key for 1
#     else:
#         prediction = f"It's a {list(class_indices.keys())[0]}"  # Key for 0

#     st.write(prediction)

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Add dynamic background and color styling

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
    }
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    footer {
        text-align: center;
        color: gray;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title with emojis
st.title("🚗🏍️ Car or Bike Identifier 🚗🏍️")
st.subheader("🔍 Upload an image and let AI do the guessing!")

# Load the model and class indices
model = tf.keras.models.load_model("car_bike_classifier.h5")
class_indices = np.load("class_indices.npy", allow_pickle=True).item()

# Sidebar feedback option
st.sidebar.header("💡 Feedback")
feedback = st.sidebar.text_area("Share your thoughts or issues here:")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback! 😊")

with st.sidebar:
    st.title("About This App")
    st.write(
        "This AI-powered tool, created by Faraz, helps users identify whether an image is of a car or a bike. It's designed for quick, easy, and accurate vehicle classification."
    )
    st.header("How It Works:")
    st.write(
        """
        1. Upload an image of a car or a bike.
        2. Click **Submit** to analyze the image.
        3. Receive instant predictions on whether it's a car or a bike.
        """
    )
    faqs = {
        "Problem Statement": "Identifying vehicles quickly is often required in various fields such as traffic analysis, insurance, or even in entertainment applications. This tool offers a simple and quick way to distinguish between cars and bikes.",
        "Solution Overview": "This tool leverages deep learning to analyze images and classify them as either a car or a bike. It uses a pre-trained model to deliver fast and reliable results for vehicle identification."
    }
    for question, answer in faqs.items():
        with st.expander(question):
            st.write(answer)

# File uploader
uploaded_file = st.file_uploader("Upload an image of a Car or Bike", type=["jpg", "png", "jpeg"])

# Slider for confidence threshold
# Show example predictions in the sidebar
# st.sidebar.header("📸 Example Predictions")
# st.sidebar.image("car_example.jpg", caption="Detected: Car")
# st.sidebar.image("bike_example.jpg", caption="Detected: Bike")

# Prediction logic
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    test_image = load_img(uploaded_file, target_size=(64, 64))
    test_image_array = img_to_array(test_image) / 255.0
    test_image_expanded = np.expand_dims(test_image_array, axis=0)

    # Show a loading spinner
    with st.spinner("🤖 Analyzing your image..."):
        result = model.predict(test_image_expanded)

    # Get prediction confidence

    # Predict based on confidence threshold
            # Show the prediction
    if result[0][0] < 0.5:
        prediction = f"🏍️ It's a {list(class_indices.keys())[0]}! 🏍️"
        st.image("bike.gif")  # Use a bike GIF
    else:
        prediction = f"🚗 It's a {list(class_indices.keys())[1]}! 🚗"
        st.image("car.gif")  # Use a car GIF

    # Display prediction result
    st.write(prediction)