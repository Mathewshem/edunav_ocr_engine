import os
import torch
from torch import nn
from torchvision import models
from src.utils import project_root, load_labels

def export_onnx(
    model_dir=os.path.join(project_root(), "models"),
    out_name="mobilenet_v2_best.onnx"
):
    labels = load_labels(os.path.join(model_dir, "labels.json"))
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(labels))
    state_path = os.path.join(model_dir, "mobilenet_v2_best.pt")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(model_dir, out_name)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12
    )
    print(f"✅ Exported ONNX to {onnx_path}")

if __name__ == "__main__":
    export_onnx()
