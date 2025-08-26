import os
import cv2
import torch
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from ultralytics import YOLO  # ✅ Dùng mô hình mới của ultralytics

from utils_LP import crop_n_rotate_LP

# Cấu hình môi trường
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Thông số
image_path = 'data/test/images/test25.jpg'
LP_weights_path = 'license_plate_detector.pt'  # ✅ Thay thế bằng mô hình .pt mới

# Khởi tạo PaddleOCR
ocr = PaddleOCR(lang='en')

# Load mô hình YOLO mới
model_LP = YOLO(LP_weights_path)

# Đọc ảnh gốc
source_img = cv2.imread(image_path)
assert source_img is not None, f"Không thể đọc ảnh từ {image_path}"

# Phát hiện biển số bằng model YOLO mới
results = model_LP.predict(source_img, conf=0.25)

# Chuyển kết quả sang định dạng tương thích cũ
pred = []
for result in results:
    dets = []
    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        x1, y1, x2, y2 = box.tolist()
        dets.append([x1, y1, x2, y2, conf.item(), cls.item()])
    pred.append(torch.tensor(dets))

# Dùng ảnh gốc để hiển thị kết quả
LP_detected_img = source_img.copy()

# Xử lý từng biển số được phát hiện
c = 0
for det in pred:
    for *xyxy, conf, cls in reversed(det):
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Cắt và xoay biển số
        angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(source_img, x1, y1, x2, y2)
        if rotate_thresh is None or LP_rotated is None:
            continue

        # Hiển thị ảnh biển số đã xoay
        cv2.imshow(f'Cropped and Rotated License Plate {c}', LP_rotated)

        # Chuyển sang RGB và OCR
        LP_rotated_rgb = cv2.cvtColor(LP_rotated, cv2.COLOR_BGR2RGB)
        LP_rotated_pil = Image.fromarray(LP_rotated_rgb)

        # OCR nhận dạng
        result = ocr.ocr(np.array(LP_rotated_pil), cls=True)

        # Trích xuất và hiển thị kết quả
        recognized_text = ' '.join([line[1][0] for line in result[0]])
        print('Recognized License Plate:', recognized_text)

        # Ghi kết quả lên ảnh
        cv2.putText(LP_detected_img, recognized_text, (x1, y1 - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
        c += 1

print('✅ Hoàn tất xử lý!')
cv2.imshow('Detected license plates', cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5))
cv2.waitKey(0)
