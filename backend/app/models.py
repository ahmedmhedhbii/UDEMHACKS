from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from transformers import pipeline
import PyPDF2

# For image processing:
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

router = APIRouter(tags=["doctor-llm"])

# ----------------------------------------------------------------------
# Initialize the Medical LLM pipeline (for text & PDF inputs)
# ----------------------------------------------------------------------
try:
    medical_llm = pipeline(
    "text-generation",
    model="openlifescienceai/open_medical_llm_leaderboard",
    use_safetensors=True,
    trust_remote_code=True,
    device=0  
    )

except Exception as e:
    raise RuntimeError(f"Failed to load open_medical_llm_leaderboard model: {e}")


try:
    chexnet_model = load_model("path/to/chexnet_model.h5")
except Exception as e:
    raise RuntimeError(f"Failed to load CheXNet model: {e}")

# ----------------------------------------------------------------------
# Pydantic models for API requests and responses
# ----------------------------------------------------------------------
class LLMRequest(BaseModel):
    input_text: str

class LLMResponse(BaseModel):
    result: str

class CheXNetResponse(BaseModel):
    predictions: list

# ----------------------------------------------------------------------
# Utility: Extract text from a PDF file using PyPDF2.
# ----------------------------------------------------------------------
def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    try:
        # Ensure the file pointer is at the beginning.
        pdf_file.file.seek(0)
        reader = PyPDF2.PdfReader(pdf_file.file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF file: {e}")

# ----------------------------------------------------------------------
# Endpoint 1: Medical LLM for text input (with optional PDF context)
# ----------------------------------------------------------------------
@router.post("/use-llm", response_model=LLMResponse)
async def doctor_use_llm(
    request: LLMRequest,
    pdf: UploadFile = File(None)
):
    """
    Endpoint for querying the medical LLM.
    Optionally, upload a PDF to extract text and add it as additional context.
    """
    combined_input = request.input_text

    if pdf is not None:
        pdf_text = extract_text_from_pdf(pdf)
        combined_input += "\n\nAdditional context from PDF:\n" + pdf_text

    try:
        # Adjust parameters (like max_length) as needed.
        output = medical_llm(combined_input, max_length=300, do_sample=True)
        result_text = output[0]['generated_text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM inference error: {str(e)}")
    
    return LLMResponse(result=result_text)

# ----------------------------------------------------------------------
# Endpoint 2: CheXNet for image analysis
# ----------------------------------------------------------------------
@router.post("/chexnet", response_model=CheXNetResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyzes a chest X-ray image using the CheXNet-Keras model.
    The image is expected to be an RGB image; it is resized to 224x224 (adjust if necessary).
    """
    try:
        # Open and preprocess the image.
        image = Image.open(file.file).convert("RGB")
        image = image.resize((224, 224))  # Resize as required by the CheXNet model.
        img_array = np.array(image) / 255.0  # Normalize pixel values.
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension.

        preds = chexnet_model.predict(img_array)
        # Optionally, apply postprocessing to the predictions here.
        return CheXNetResponse(predictions=preds.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")
