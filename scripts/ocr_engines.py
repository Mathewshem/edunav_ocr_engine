# scripts/ocr_engines.py
import os
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# If you need Windows path to tesseract, you can keep it optional:
TES_PATH = os.environ.get("TESSERACT_EXE", "")
if TES_PATH:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Optional: your existing OpenCV preprocessing
try:
    from src.preprocess import preprocess_image  # if you already have this
    _HAS_PRE = True
except Exception:
    _HAS_PRE = False

# ---- TrOCR (handwritten) ----
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Use a small model for CPU/Pi. This is the handwritten variant.
_TROCR_NAME = "microsoft/trocr-small-handwritten"

_processor = None
_model     = None

def _load_trocr_once():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = TrOCRProcessor.from_pretrained(_TROCR_NAME)
        _model = VisionEncoderDecoderModel.from_pretrained(_TROCR_NAME)
        _model.eval()

def read_printed(image_path: str) -> str:
    if _HAS_PRE:
        # If your preprocess returns a numpy array, save to temp PIL:
        pil = Image.open(image_path)
        text = pytesseract.image_to_string(pil)
    else:
        pil = Image.open(image_path)
        text = pytesseract.image_to_string(pil)
    return text.strip()

@torch.no_grad()
def read_handwritten(image_path: str) -> str:
    _load_trocr_once()
    image = Image.open(image_path).convert("RGB")
    pixel_values = _processor(images=image, return_tensors="pt").pixel_values
    generated_ids = _model.generate(pixel_values, max_length=128)
    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


