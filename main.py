import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Load the YOLO model
model = YOLO("./static/models/yolo/best.pt")  # Replace with your model path

# Streamlit app
st.title("YOLO Logo Detection and Blurring")

# Sidebar for uploading image
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
run_button = st.sidebar.button("Detect and Blur Logos")
image_placeholder = st.empty()

# State management
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

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # xyxy format: (x1, y1, x2, y2)
        blurred_image = image_array.copy()

        # Apply blur to each detected box
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = blurred_image[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (1001, 1001), 0)
            blurred_image[y1:y2, x1:x2] = blurred_roi

        # Optional: Draw boxes on the blurred image

        st.session_state.annotated_image = blurred_image
        st.success("Logos blurred successfully!")
        image_placeholder.image(st.session_state.annotated_image, caption="Blurred Detection Results", channels="RGB")

else:
    st.write("Upload an image to get started.")
