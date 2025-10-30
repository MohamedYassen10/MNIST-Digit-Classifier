# MNIST Digit Classifier — FastAPI

A simple REST API that classifies handwritten digits using a Keras CNN trained on MNIST.
Upload an image (PNG/JPG) and the API returns the predicted digit and confidence.

## Features
- Lightweight FastAPI app
- Uses Pillow for image loading and preprocessing
- Preprocessing: grayscale -> resize 28x28 -> normalize -> reshape
- No color inversion (matches model training)
- Returns `{predicted_digit, confidence}` JSON

## Files
- `Api.py` — FastAPI server
- `app.py` — streamlit application
- `model/robust_mnist_final.keras` — pre-trained Keras model (put here)
- `requirements.txt` — Python dependencies
