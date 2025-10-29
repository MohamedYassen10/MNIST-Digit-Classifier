import streamlit as st 
from PIL import Image
import requests
API_URL = "http://localhost:8080/predict_digit/"
st.title("Streamlit UI for FastAPI MNIST API")
st.markdown("Upload an image of a handwritten digit (0-9) to get its prediction from the FastAPI MNIST API.")
uploaded_file=st.file_uploader("Choose an image...",type=['PNG','JPG','JPEG'])

if uploaded_file is not None :
    image = Image.open(uploaded_file)
    st.image(image,caption='Image for Prediction',use_column_width=True)
    with st.spinner("sending image to Fast API for prediction..."):
        files={'file':uploaded_file.getvalue()}
        try:
            respond = requests.post(API_URL, files=files)
            if respond.status_code == 200:
                prediction = respond.json()
                st.header("API Prediction Results")
                st.success(f"Predicted Digit: {prediction['predicted_digit']}")
                st.info(f"Confidence: {prediction['confidence']:.4f}")
            else:
                st.error(f"API Error: {respond.status_code}-{respond.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI server. Please ensure the server is running on http://localhost:8080")
    
else :
    st.info("Please upload an image file to get started.")
    

