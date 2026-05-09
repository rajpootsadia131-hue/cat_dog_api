from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os
import gdown

app = FastAPI()

# Model download karne ka code - abhi link baad mein daalenge
MODEL_PATH = "cats_dogs_vgg16_84percent.h5"
if not os.path.exists(MODEL_PATH):
    # Yahan Google Drive link daalna hai - abhi khaali chhod do
    gdown.download("3DUDY2AMdrFjkLrVevCHgU0EhDt_38hMSXTgDyNeucXAaJbzF", MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Cat vs Dog Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    result = "Dog" if prediction[0][0] > 0.5 else "Cat"
    confidence = float(prediction[0][0]) if result == "Dog" else float(1 - prediction[0][0])
    
    return JSONResponse({"prediction": result, "confidence": confidence})
