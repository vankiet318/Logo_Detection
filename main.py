import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Load the YOLO model
model = YOLO("./static/models/yolo/best.pt")  # Replace with the path to your trained model

# Streamlit app
st.title("YOLO Object Detection")

# Sidebar for uploading image
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
run_button = st.sidebar.button("Detect logo")
image_placeholder = st.empty()

# State management to store detection result
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

if uploaded_file:
	image = Image.open(uploaded_file)
	image_placeholder.image(image, caption="Uploaded Image")
	if run_button:
		image_array = np.array(image)
		if image_array.shape[-1] == 4:
			image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

		with st.spinner("Running YOLO detection..."):
			results = model(image_array)

		annotated_image = results[0].plot()

		st.session_state.annotated_image = annotated_image
		st.success("Detection complete!")
		image_placeholder.image(st.session_state.annotated_image, caption="Detection Results")
else:
    st.write("Upload an image to get started.")
