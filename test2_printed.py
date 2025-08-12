#import pytesseract

# Explicit path to tesseract.exe
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#import pytesseract
#print(pytesseract.get_tesseract_version())


import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from scripts.ocr_engines import read_printed
print(read_printed(r"C:\Users\USER\Desktop\Hackathon25\STEM\models\ocr_engine\data\test\printed\image 31.png")[:120])
