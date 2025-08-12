#python - <<'PY'
from PIL import Image
from src.infer import OCRClassifier
clf = OCRClassifier()
img = Image.open(r"C:\Users\USER\Desktop\Hackathon25\STEM\models\ocr_engine\data\test\printed\image 31.png").convert("RGB")
print(clf.classify(img))

