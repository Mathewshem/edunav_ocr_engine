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



### 2) Raspberry Pi (inference)
- Install OpenCV:
sudo apt-get update && sudo apt-get install -y python3-opencv

- Use ONNX Runtime for inference:
pip3 install onnxruntime --break-system-packages

- If Torch is heavy on Pi 2GB, skip it.

## Folders
- `data/` → train/val/test splits in three classes: Printed, Handwritten, Diagram
- `models/` → trained files: `mobilenet_v2_best.pt`, `mobilenet_v2_best.onnx`, `labels.json`
- `src/` → training, export, inference code
- `api/` → FastAPI app

## Train (Windows)
python -m src.train



## Export to ONNX (for Pi)
python -m src.export_onnx


## Run API (Windows or Pi)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000


Open Swagger: `http://<host>:8000/docs` → `POST /ocr-classify`

## Output JSON
{
"label": "Printed",
"confidence": 0.91,
"engine": "pytesseract",

EduNav+ OCR Classifier & Auto-OCR API
An OCR microservice for EduNav+ that:

Classifies images as printed / handwritten / diagram

Routes to the right engine:

Printed → Tesseract (pytesseract)

Handwritten → TrOCR (Hugging Face) (fallback to Tesseract if unavailable)

Diagram → placeholder (future handler)

Optionally speaks extracted text on the host device (Pi/PC)

Features
FastAPI REST API with Swagger docs

Transfer-learning classifier (MobileNetV2) exported to ONNX for Raspberry Pi

Robust OCR routing with fallbacks & error handling

Works on Windows (dev) and Raspberry Pi 4 (2GB)

Project Structure (simplified)

models/ocr_engine/
├─ api/
│  └─ main.py                # FastAPI app (/ocr-classify, /ocr-auto)
├─ scripts/
│  ├─ ocr_engines.py         # read_printed (Tesseract), read_handwritten (TrOCR)
│  ├─ tts_engine.py          # speak_text() using espeak/pyttsx3/gTTS
│  └─ classifier.py          # (optional) simple PIL wrapper
├─ src/
│  ├─ train.py               # train classifier (MobileNetV2)
│  ├─ export_onnx.py         # export to ONNX
│  ├─ infer.py               # OCRClassifier class (ONNX or Torch)
│  ├─ preprocess.py          # image transforms
│  └─ utils.py               # labels, paths, seeds
├─ models/
│  ├─ mobilenet_v2_best.pt
│  ├─ mobilenet_v2_best.onnx
│  └─ labels.json            # ["printed","handwritten","diagram"]
├─ requirements.txt
└─ README.md
Quick Start (Windows dev)
Create & activate venv, install deps


pip install -r requirements.txt
# TrOCR (handwritten) extras:
pip install transformers sentencepiece timm accelerate protobuf
(Windows only) Point pytesseract to Tesseract.exe
Edit scripts/ocr_engines.py if Tesseract isn’t on PATH:


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
Run API


cd models/ocr_engine
uvicorn api.main:app --host 0.0.0.0 --port 8000
Open docs:
http://127.0.0.1:8000/docs

Quick Start (Raspberry Pi 4B, 2GB)
System packages (OCR, audio, OpenCV runtime)


sudo apt update
sudo apt install -y tesseract-ocr espeak-ng python3-opencv
Python deps


pip3 install fastapi uvicorn[standard] pillow pytesseract onnxruntime --break-system-packages
# Optional for handwritten TrOCR (heavier on Pi):
pip3 install transformers sentencepiece timm accelerate protobuf --break-system-packages
Run API


cd ~/models/ocr_engine
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Open from your laptop on same Wi-Fi:
http://<PI_IP>:8000/docs

Tip: If audio doesn’t play, test: espeak "Hello from Pi" and select output device with raspi-config or PulseAudio.

Endpoints
GET /
Health & info.


{ "message": "EduNav+ OCR Classifier is running.", "endpoint": "/ocr-classify" }
POST /ocr-classify
Classify image only.

form-data: file (image)

200 response:


{
  "label": "printed|handwritten|diagram",
  "confidence": 0.91,
  "engine": "pytesseract|trocr|diagram",
  "low_confidence": false
}
POST /ocr-auto
Classify → route OCR → (optionally) speak text.

form-data: file (image)

200 response:


{
  "class": "printed",
  "confidence": 0.9321,
  "engine": "pytesseract",
  "text": "extracted text here"
}
cURL example

curl -X POST "http://<BASE_URL>/ocr-auto" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/sample.jpg"
Training & Export (done on Windows)
Train:


python -m src.train
Export ONNX (for Pi):


python -m src.export_onnx
Commit:

models/mobilenet_v2_best.onnx
models/labels.json
Environment Notes
Tesseract (Windows):

Add to PATH or set pytesseract.pytesseract.tesseract_cmd.

Handwritten (TrOCR):

Requires transformers + sentencepiece (and is heavier).

If unavailable, service falls back to Tesseract automatically.

Audio (TTS):

Pi: uses espeak-ng or pyttsx3 backend.

Windows: pyttsx3 or gTTS can be used.

You can disable speaking by no-op in scripts/tts_engine.py.

Troubleshooting
500 + “Tesseract not found”
Set Windows path in ocr_engines.py or add Tesseract to PATH.

TrOCR error mentioning SentencePiece
pip install sentencepiece protobuf

No audio output on Pi
Test espeak "hello". Select sink in pavucontrol.
For HDMI/jack: sudo raspi-config → Audio.

ONNX not loading on Pi
Ensure onnxruntime installed; fallback to Torch is supported.

Port already in use
Stop previous server or change port: --port 8001.

Roadmap
Diagram handler (alt-text generation & STEM chart narration)

Live webcam OCR endpoint (/ocr-live)

Whisper-based /listen for voice commands

Cloud deploy (Render/Railway) + ngrok for testing
"low_confidence": false
}

