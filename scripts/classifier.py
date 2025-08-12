# scripts/classifier.py
import os, json
import numpy as np
import onnxruntime as ort
from PIL import Image

# Resolve models folder relative to this file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "models"))

# Files produced by your export step
ONNX_PATH   = os.path.join(_MODELS_DIR, "mobilenet_v2_best.onnx")
LABELS_PATH = os.path.join(_MODELS_DIR, "labels.json")

# Load once
_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
_labels  = json.load(open(LABELS_PATH, "r"))

# Same preprocessing used during training/export
def _preprocess_pil(img: Image.Image, size=224):
    img = img.convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    # torchvision normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))          # CHW
    arr = np.expand_dims(arr, 0)                # NCHW
    return arr

def classify_pil(img: Image.Image):
    inp = _preprocess_pil(img)
    ort_inputs = {_session.get_inputs()[0].name: inp}
    logits = _session.run(None, ort_inputs)[0][0]  # (num_classes,)
    # softmax
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    idx = int(np.argmax(probs))
    return _labels[str(idx)], float(probs[idx])
