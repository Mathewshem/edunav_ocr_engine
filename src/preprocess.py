from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

IMG_SIZE = 224

# For training: augmentation + normalization
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.05)], p=0.5),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

# For validation/inference
def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

# Optional denoise/gray step before PIL (useful for messy images)
def cv2_pre_denoise(pil_image: Image.Image) -> Image.Image:
    img = np.array(pil_image)
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # mild denoise + threshold
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
    # stack to 3 channels for MobileNet
    stacked = np.stack([denoised]*3, axis=-1)
    return Image.fromarray(stacked)
