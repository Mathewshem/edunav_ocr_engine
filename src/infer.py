import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn
from src.preprocess import get_eval_transform, cv2_pre_denoise
from src.utils import load_labels, project_root

# Try ONNX first (for Pi), fallback to Torch
try:
    import onnxruntime as ort
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False

def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

class OCRClassifier:
    def __init__(self,
                 model_dir=os.path.join(project_root(), "models"),
                 prefer_onnx=True):
        self.model_dir = model_dir
        self.labels = load_labels(os.path.join(model_dir, "labels.json"))
        self.eval_tf = get_eval_transform()

        self.session = None
        self.torch_model = None

        if prefer_onnx and HAS_ONNX and os.path.exists(os.path.join(model_dir, "mobilenet_v2_best.onnx")):
            onnx_path = os.path.join(model_dir, "mobilenet_v2_best.onnx")
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        else:
            # Torch fallback
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, len(self.labels))
            state = torch.load(os.path.join(model_dir, "mobilenet_v2_best.pt"), map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self.torch_model = model

    def classify(self, image: Image.Image):
        # optional denoise
        image = cv2_pre_denoise(image)
        tensor = self.eval_tf(image).unsqueeze(0)  # [1,3,224,224]

        if self.session is not None:
            logits = self.session.run(["logits"], {"input": tensor.numpy()})[0]  # [1,C]
            probs = _softmax(logits)[0]
        else:
            with torch.no_grad():
                logits = self.torch_model(tensor)
                probs = F.softmax(logits, dim=1).numpy()[0]

        idx = int(np.argmax(probs))
        label = self.labels[idx]
        conf = float(probs[idx])
        # routing hint
        if label.lower() == "printed":
            engine = "pytesseract"
        elif label.lower() == "handwritten":
            engine = "trocr"
        else:
            engine = "diagram"

        return {"label": label, "confidence": conf, "engine": engine}

# CLI test (optional)
if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]
    clf = OCRClassifier()
    out = clf.classify(Image.open(img_path).convert("RGB"))
    print(out)
