# src/infer.py
import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from src.utils import load_labels, project_root

# Optional preprocess imports
try:
    from src.preprocess import get_eval_transform, cv2_pre_denoise
    HAS_PRE = True
except Exception:
    HAS_PRE = False
    def get_eval_transform():
        # minimal safe fallback
        from torchvision import transforms as T
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    def cv2_pre_denoise(img_pil):
        return img_pil

# Try ONNX first
try:
    import onnxruntime as ort
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False


def _softmax_np(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class OCRClassifier:
    def __init__(self,
                 model_dir=os.path.join(project_root(), "models"),
                 prefer_onnx=True):
        self.model_dir = model_dir

        # --- Labels (must be a list) ---
        labels_path = os.path.join(model_dir, "labels.json")
        self.labels = load_labels(labels_path)
        if not isinstance(self.labels, list):
            raise RuntimeError(
                f"labels.json must be a list. Got: {type(self.labels)}"
            )

        self.eval_tf = get_eval_transform()

        self.session = None
        self.torch_model = None
        self._onnx_input_name = None
        self._onnx_output_name = None

        onnx_file = os.path.join(model_dir, "mobilenet_v2_best.onnx")
        pt_file   = os.path.join(model_dir, "mobilenet_v2_best.pt")

        if prefer_onnx and HAS_ONNX and os.path.exists(onnx_file):
            # ---- ONNX path ----
            self.session = ort.InferenceSession(
                onnx_file,
                providers=["CPUExecutionProvider"]
            )
            # auto-detect names (critical to avoid str-index bugs)
            self._onnx_input_name = self.session.get_inputs()[0].name
            self._onnx_output_name = self.session.get_outputs()[0].name
        else:
            # ---- Torch path ----
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, len(self.labels))
            if not os.path.exists(pt_file):
                raise FileNotFoundError(f"Missing model file: {pt_file}")
            state = torch.load(pt_file, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self.torch_model = model

    def classify(self, image: Image.Image):
        """
        image: PIL.Image (RGB)
        returns: dict {label, confidence, engine}
        """
        # optional denoise
        image = cv2_pre_denoise(image)

        x = self.eval_tf(image).unsqueeze(0)  # [1,3,224,224]

        if self.session is not None:
            # ONNX
            logits = self.session.run(
                [self._onnx_output_name],
                {self._onnx_input_name: x.numpy()}
            )[0]  # -> [1, C]
            probs = _softmax_np(logits)[0]
        else:
            # Torch
            with torch.no_grad():
                logits = self.torch_model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        # ✅ self.labels must be LIST → safe integer index
        label = self.labels[idx]

        # engine hint for routing
        if label.lower() == "printed":
            engine = "pytesseract"
        elif label.lower() == "handwritten":
            engine = "trocr"
        else:
            engine = "diagram"

        return {"label": label, "confidence": conf, "engine": engine}


# quick CLI test
if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]
    clf = OCRClassifier()
    out = clf.classify(Image.open(img_path).convert("RGB"))
    print(out)
