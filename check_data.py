import cv2
import os
import numpy as np
import random

# --- CẤU HÌNH ---
# Kiểm tra tập train hoặc val tùy bạn chọn
IMAGES_PATH = 'datasets/train/images'
LABELS_PATH = 'datasets/train/labels'

def draw_yolo_labels():
    image_files = [f for f in os.listdir(IMAGES_PATH) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Chọn ngẫu nhiên 5 ảnh để kiểm tra nhanh
    sample_images = random.sample(image_files, min(5, len(image_files)))

    for img_name in sample_images:
        img_path = os.path.join(IMAGES_PATH, img_name)
        label_path = os.path.join(LABELS_PATH, os.path.splitext(img_name)[0] + '.txt')

        if not os.path.exists(label_path):
            print(f"Cảnh báo: Không tìm thấy nhãn cho {img_name}")
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_id = parts[0]
            # Tọa độ YOLO (0 -> 1)
            cx, cy, bw, bh = map(float, parts[1:])

            # Chuyển ngược về tọa độ Pixel
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # Vẽ khung chữ nhật (Màu đỏ để dễ nhìn)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Class: {class_id}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Hiển thị kết quả
        cv2.imshow(f"Kiểm tra nhãn: {img_name}", img)
        print(f"Đang hiển thị {img_name}. Nhấn phím bất kỳ để xem ảnh tiếp theo...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    draw_yolo_labels()