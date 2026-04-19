import pytesseract
from PIL import Image

img = Image.open("test.png")  # put any image with text
print(pytesseract.image_to_string(img))