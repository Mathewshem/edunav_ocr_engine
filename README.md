# EduNav+ OCR Classifier (Printed vs Handwritten vs Diagram)

Routes an input image to the correct OCR engine:
- Printed → pytesseract
- Handwritten → TrOCR (transformer OCR)
- Diagram → placeholder / future handler

## Setup

### 1) Install requirements (Windows / training)

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

markdown
Copy
Edit

### 2) Raspberry Pi (inference)
- Install OpenCV:
sudo apt-get update && sudo apt-get install -y python3-opencv

diff
Copy
Edit
- Use ONNX Runtime for inference:
pip3 install onnxruntime --break-system-packages

markdown
Copy
Edit
- If Torch is heavy on Pi 2GB, skip it.

## Folders
- `data/` → train/val/test splits in three classes: Printed, Handwritten, Diagram
- `models/` → trained files: `mobilenet_v2_best.pt`, `mobilenet_v2_best.onnx`, `labels.json`
- `src/` → training, export, inference code
- `api/` → FastAPI app

## Train (Windows)
python -m src.train

graphql
Copy
Edit

## Export to ONNX (for Pi)
python -m src.export_onnx

shell
Copy
Edit

## Run API (Windows or Pi)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

bash
Copy
Edit
Open Swagger: `http://<host>:8000/docs` → `POST /ocr-classify`

## Output JSON
{
"label": "Printed",
"confidence": 0.91,
"engine": "pytesseract",
"low_confidence": false
}

Copy
Edit
