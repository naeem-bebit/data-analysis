from paddleocr import PaddleOCR, draw_ocr  # main OCR dependencies
from matplotlib import pyplot as plt  # plot images
import cv2  # opencv
import os

img_path = os.path.join('.', 'sample_data', 'sample_pdf.jpg')
result = ocr_model.ocr(img_path)

for res in result:
    print(res[1][0])

boxes = [res[0] for res in result]
texts = [res[1][0] for res in result]
scores = [res[1][1] for res in result]
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
font = './sample_data/simfang.ttf'
plt.figure(figsize=(15, 15))
annotated = draw_ocr(img, boxes, texts, scores, font_path=font)
plt.imshow(annotated)
