import io
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


router = APIRouter(tags=["doctor-radiology"])

# Load your CheXNet Keras model.
# Update the path below to where your model file is saved.
try:
    model_path = "brucechou1983_CheXNet_Keras_0.3.0_weights.h55"  # Replace with your model file path
    chexnet_model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load CheXNet model: {e}")

# Define a response model
class RadiologyResponse(BaseModel):
    predictions: dict  # For example, a dictionary with disease names and probabilities

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image to match the CheXNet requirements.
    Typically, CheXNet expects a 224x224 image.
    """
    try:
        # Resize the image to 224x224 and convert it to RGB if needed.
        image = image.resize((224, 224)).convert("RGB")
        image_array = img_to_array(image)
        # Normalize the image (CheXNet was trained on images normalized to [0, 1] or using ImageNet statistics)
        image_array = image_array / 255.0
        # Expand dimensions to create a batch of size 1
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

@router.post("/analyze-xray", response_model=RadiologyResponse)
async def analyze_chest_xray(
    image: UploadFile = File(...),
):
    """
    Endpoint for doctors to upload a chest X-ray image and get disease predictions.
    Uses CheXNet-Keras to detect pneumonia (and potentially other pathologies).
    """
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing uploaded image: {e}")

    # Preprocess the image
    input_image = preprocess_image(pil_image)

    try:
        # Run inference using CheXNet model
        preds = chexnet_model.predict(input_image)[0]
        # Assuming the model outputs probabilities for a fixed list of classes.
        # Replace the labels with those used in your CheXNet model.
        labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
                  "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", 
                  "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
        predictions = {label: float(pred) for label, pred in zip(labels, preds)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    return RadiologyResponse(predictions=predictions)
