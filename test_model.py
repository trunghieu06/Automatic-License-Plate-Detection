import cv2
import os
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
VAL_IMAGES_PATH = 'datasets/test'
MODEL_PATH = 'runs/detect/alpr_v1_6gb_vram/weights/best.pt'
OUTPUT_DIR = 'val_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- KHỞI TẠO ---
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])

def show_preprocessing_steps(plate_crop, gray, thresh):
    print("Show")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("1. Original Crop")
    plt.imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
    plt.axis('off') 

    plt.subplot(1, 3, 2)
    plt.title("2. Grayscale")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("3. Threshold")
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def improve_ocr_input(plate_crop):
    """Tiền xử lý nâng cao: Nắn thẳng, khử nhiễu, phân ngưỡng."""
    # 1. Chuyển xám và khử nhiễu
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Tìm cạnh và nắn thẳng (Perspective Transform)
    edged = cv2.Canny(blurred, 30, 150)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    refined_plate = gray.copy()
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # Sắp xếp điểm và thực hiện biến đổi phối cảnh
            rect = np.zeros((4, 2), dtype="float32")
            s = approx.sum(axis=2); rect[0] = approx[np.argmin(s)]; rect[2] = approx[np.argmax(s)]
            diff = np.diff(approx, axis=2); rect[1] = approx[np.argmin(diff)]; rect[3] = approx[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            width = max(int(np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))), int(np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))))
            height = max(int(np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))), int(np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))))
            dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            refined_plate = cv2.warpPerspective(gray, M, (width, height))
            break

    # 3. Phóng to và Thresholding
    refined_plate = cv2.resize(refined_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(refined_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    
    # 4. Làm đậm nét chữ (Morphological)
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    show_preprocessing_steps(plate_crop, gray, thresh)

    return thresh

def reduce_glare_for_yolo(image_bgr):
    """
    Tiền xử lý ảnh toàn cục: Khôi phục chi tiết vùng chói sáng 
    trước khi đưa vào YOLO nhận diện.
    """
    # 1. Chuyển đổi từ BGR sang không gian màu LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    
    # 2. Tách các kênh L (Lightness), A, B
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 3. Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) lên kênh L
    # clipLimit=3.0 giúp kéo lại các vùng sáng gắt rất hiệu quả
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)
    
    # 4. Gộp kênh L đã xử lý với các kênh màu cũ
    merged_lab = cv2.merge((l_clahe, a_channel, b_channel))
    
    # 5. Chuyển ngược lại về BGR cho YOLO
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

def core_pipeline(image_bgr):
    """Hàm lõi: Nhận vào ảnh BGR -> Vẽ khung và viết TEXT trực tiếp lên ảnh."""
    
    # BƯỚC MỚI: Dập tắt chói sáng trước khi cho YOLO nhìn
    enhanced_image = reduce_glare_for_yolo(image_bgr)
    
    # Đưa ảnh đã dập chói vào YOLO (vẫn có thể giữ conf=0.25 hoặc 0.5 tùy bạn)
    results = model(enhanced_image, conf=0.25, verbose=False)[0] 
    
    detected_plates = []
    
    for box in results.boxes:
        # 1. Lấy tọa độ và độ tự tin
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        
        # 2. Tiền xử lý và OCR (Sử dụng hàm improve_ocr_input đã viết)
        plate_crop = image_bgr[y1:y2, x1:x2]
        processed_plate = improve_ocr_input(plate_crop)
        
        ocr_results = reader.readtext(processed_plate, allowlist='0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')
        text = "".join([res[1].upper() for res in ocr_results])
        text = "".join(re.findall(r'[A-Z0-9]', text)) 
        
        detected_plates.append(text)
        
        # --- BẮT ĐẦU VẼ LÊN ẢNH ---
        # 3. Vẽ khung hình chữ nhật cho biển số (Màu xanh lá)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 4. Tạo nền cho văn bản (giúp chữ dễ đọc hơn trên mọi nền ảnh)
        label = f"{text} ({conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Vẽ một hình chữ nhật đặc phía trên khung biển số để làm nền cho chữ
        cv2.rectangle(image_bgr, (x1, y1 - h - 15), (x1 + w, y1), (0, 255, 0), -1)
        
        # 5. Viết văn bản lên nền vừa tạo (Chữ màu đen trên nền xanh)
        cv2.putText(
            image_bgr, 
            label, 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8,              # Font size
            (0, 0, 0),        # Màu chữ (Đen)
            2,                # Độ dày nét chữ
            cv2.LINE_AA       # Khử răng cưa giúp chữ mượt hơn
        )
        
    return image_bgr, detected_plates

if __name__ == "__main__":
    # run_val_pipeline()
    img_path = 'datasets/test/17.jpg'
    # print(img_path)
    img = cv2.imread(img_path)
    core_pipeline(img)
    # process_video()