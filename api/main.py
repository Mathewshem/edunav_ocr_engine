from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import traceback
import os

from src.infer import OCRClassifier
from src.utils import project_root

app = FastAPI(title="EduNav+ OCR Classifier")

# Load model at startup (ONNX preferred on Pi)
MODEL = None

@app.on_event("startup")
def _load_model():
    global MODEL
    MODEL = OCRClassifier(
        model_dir=os.path.join(project_root(), "models"),
        prefer_onnx=True
    )

@app.get("/")
def root():
    return {"message": "EduNav+ OCR Classifier is running.", "endpoint": "/ocr-classify"}

@app.post("/ocr-classify")
async def ocr_classify(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        result = MODEL.classify(image)
        # add confidence threshold hint
        result["low_confidence"] = (result["confidence"] < 0.65)
        return JSONResponse(result)
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)
