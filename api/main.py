
import os
import sys
import io
import shutil
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# ---- ensure package root on sys.path ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# ---- project imports ----
from src.infer import OCRClassifier
from src.utils import project_root
from scripts.ocr_engines import read_printed   # we will NOT call read_handwritten until sentencepiece installed
from scripts.tts_engine import speak_text      # uses your espeak/pyttsx3 fallback

app = FastAPI(title="EduNav+ OCR Classifier")

# ---- load model once (ONNX preferred) ----
clf = None

@app.on_event("startup")
def _load_model():
    global clf
    clf = OCRClassifier(
        model_dir=os.path.join(project_root(), "models"),
        prefer_onnx=True
    )

@app.get("/")
def root():
    return {"message": "EduNav+ OCR is running", "endpoints": ["/ocr-classify", "/ocr-auto"]}

# ---------- Endpoint 1: classify only ----------
@app.post("/ocr-classify")
async def ocr_classify(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        result = clf.classify(image)  # -> {"label","confidence","engine"}
        result["low_confidence"] = (result["confidence"] < 0.65)
        return JSONResponse(result)
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)

# ---------- Endpoint 2: classify -> OCR -> (try) TTS ----------
# api/main.py
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os, shutil, tempfile
from PIL import Image

from src.infer import OCRClassifier
from scripts.ocr_engines import read_printed, read_handwritten
from scripts.tts_engine import speak_text

app = FastAPI(title="EduNav+ OCR Classifier")
clf = OCRClassifier()

@app.post("/ocr-auto")
async def ocr_auto(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        pil = Image.open(temp_path).convert("RGB")
        out = clf.classify(pil)  # <- returns dict: {"label","confidence","engine"}
        out = clf.classify(pil)
        print("DEBUG raw clf.classify output ->", out, type(out))

        # Flatten until we get a dict
        while isinstance(out, list) and len(out) > 0:
            print("DEBUG unwrapping list ->", out[0], type(out[0]))
            out = out[0]

        if not isinstance(out, dict):
            raise ValueError(f"Unexpected classify() output: {out}")

        print("DEBUG final out dict ->", out)

        pred_label = out["label"]
        conf = float(out["confidence"])
        engine = out["engine"]

        # 3) Route to correct OCR (use TrOCR for handwritten; fallback to pytesseract)
        if engine == "pytesseract":
            try:
                text = read_printed(temp_path)
            except Exception as e:
                print(f"[Tesseract warn] {e}")
                text = "[Printed OCR failed]"

        elif engine == "trocr":
            try:
                text = read_handwritten(temp_path)
                if not text.strip():
                    text = read_printed(temp_path)
            except Exception as e:
                print(f"[TrOCR warn] {e}")
                try:
                    text = read_printed(temp_path)
                except Exception as te:
                    print(f"[Fallback printed OCR warn] {te}")
                    text = "[Handwritten OCR failed]"

        else:
            try:
                text = read_printed(temp_path)
            except Exception as e:
                print(f"[Diagram fallback OCR warn] {e}")
                text = "[Diagram OCR failed]"

        try:
            if text:
                speak_text(text)
        except Exception as e:
            print(f"[TTS warn] {e}")

        return {"class": pred_label, "confidence": round(conf, 4), "engine": engine, "text": text}

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
