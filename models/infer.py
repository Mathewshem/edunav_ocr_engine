import json, os
import numpy as np
import onnxruntime as ort

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ONNX_PATH   = os.path.join(MODELS_DIR, "mobilenet_v2_best.onnx")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

with open(LABELS_PATH, "r") as f:
    _labels_raw = json.load(f)

# normalize to list: ["printed","handwritten","diagram"]
if isinstance(_labels_raw, dict):
    # assumes keys are "0","1","2",...
    LABELS = [ _labels_raw[str(i)] for i in range(len(_labels_raw)) ]
else:
    LABELS = list(_labels_raw)

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

def preprocess_for_classifier(bgr_img):
    # same preprocessing you used for training (resize, normalize, CHW…)
    import cv2
    img = cv2.resize(bgr_img, (224, 224))
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, 0..1
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (0, 1, 2))            # HWC->CHW
    img = np.expand_dims(img, 0)                   # NCHW
    return img

def classify_np(bgr_img):
    x = preprocess_for_classifier(bgr_img)
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: x})[0]  # (1,C)
    probs = softmax(logits[0])
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    return idx, conf

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)
