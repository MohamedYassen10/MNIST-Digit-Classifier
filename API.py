from fastapi import FastAPI, UploadFile, File
import numpy as np
import uvicorn
import tensorflow as tf
from io import BytesIO
from PIL import Image

# === Load trained model ===
model = tf.keras.models.load_model("robust_mnist_cnn_model.keras")


# === Image preprocessing ===
def preprocess_image(image: Image.Image):
    """
    Preprocess uploaded image to match MNIST model expectations.
    - Convert to grayscale
    - Resize to 28x28
    - Normalize to [0,1]
    - Do NOT invert colors
    """
    image = image.convert("L").resize((28, 28))       # Convert to grayscale and resize
    image_array = np.array(image).astype("float32") / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)   # Reshape for model input
    return image_array


# === Create FastAPI app ===
app = FastAPI(
    title="MNIST Digit Classifier API",
    description="An API that classifies handwritten digits using a robust CNN trained on MNIST.",
    version="1.0.0"
)


# === Root endpoint ===
@app.get("/")
def read_root():
    return {"message": "MNIST Digit Classifier API is running 🚀"}


# === Prediction endpoint ===
@app.post("/predict_digit/")
async def predict_digit(file: UploadFile = File(...)):
    """Predict the digit from the uploaded image file."""
    data = await file.read()
    image = Image.open(BytesIO(data))

    # Preprocess image
    processed_image = preprocess_image(image)

    # Predict
    prediction = model.predict(processed_image)
    predicted_digit = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4)
    }


# === Run the API ===
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
