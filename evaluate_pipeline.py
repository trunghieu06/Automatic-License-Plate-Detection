import cv2
import os
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import matplotlib.pyplot as plt
from collections import deque, Counter

# --- CẤU HÌNH ---
VAL_IMAGES_PATH = 'datasets/test'
MODEL_PATH = 'runs/detect/alpr_v1_6gb_vram/weights/best.pt'
OUTPUT_DIR = 'val_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- KHỞI TẠO ---
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])

# --- BỘ TỪ ĐIỂN VÀ LUẬT HẬU XỬ LÝ (POST-PROCESSING) ---
CHAR_TO_NUM = {'O': '0', 'D': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}

def force_number(text):
    """Ép các ký tự chữ bị nhầm lẫn thành số."""
    return ''.join([CHAR_TO_NUM.get(c, c) for c in text])

def format_vietnamese_plate(ocr_results):
    """Định dạng kết quả OCR theo luật biển số xe máy Việt Nam (2 dòng)."""
    if not ocr_results:
        return ""

    # Sắp xếp các dòng text từ trên xuống dưới theo tọa độ Y
    sorted_lines = sorted(ocr_results, key=lambda x: x[0][0][1])

    if len(sorted_lines) < 2:
        return sorted_lines[0][1].upper()

    top_line_raw = sorted_lines[0][1].upper()
    bottom_line_raw = sorted_lines[1][1].upper()

    # Xử lý dòng trên
    top_clean = re.sub(r'[^A-Z0-9]', '', top_line_raw)
    if len(top_clean) >= 4:
        province_code = force_number(top_clean[:2])
        seri_code = top_clean[2:4]
        top_final = f"{province_code}-{seri_code}"
    else:
        top_final = top_clean 

    # Xử lý dòng dưới
    bottom_clean = re.sub(r'[^A-Z0-9]', '', bottom_line_raw)
    bottom_nums = force_number(bottom_clean)

    if len(bottom_nums) == 5:
        bottom_final = f"{bottom_nums[:3]}.{bottom_nums[3:]}"
    elif len(bottom_nums) == 4:
        bottom_final = bottom_nums
    else:
        bottom_final = bottom_nums 

    return f"{top_final} {bottom_final}"


def show_preprocessing_steps(plate_crop, gray, thresh):
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
    """Sử dụng Otsu Thresholding - Giải pháp tối ưu cho ảnh đã được crop sát."""
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray_enlarged = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray_enlarged, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    final_thresh = cv2.bitwise_not(thresh)
    
    # show_preprocessing_steps(plate_crop, gray, final_thresh)
    return final_thresh

def core_pipeline(image_bgr):
    """Hàm lõi: Nhận vào ảnh BGR -> Vẽ khung và viết TEXT trực tiếp lên ảnh."""
    results = model(image_bgr, conf=0.5, verbose=False)[0]
    detected_plates = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        
        plate_crop = image_bgr[y1:y2, x1:x2]
        processed_plate = improve_ocr_input(plate_crop)
        
        # CẬP NHẬT: Thêm dấu '-' và '.' vào allowlist và gọi format_vietnamese_plate
        ocr_results = reader.readtext(processed_plate, allowlist='0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-.')
        text = format_vietnamese_plate(ocr_results)
        
        detected_plates.append(text)
        
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{text} ({conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image_bgr, (x1, y1 - h - 15), (x1 + w, y1), (0, 255, 0), -1)
        
        cv2.putText(
            image_bgr, 
            label, 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8,              
            (0, 0, 0),        
            2,                
            cv2.LINE_AA       
        )
        
    return image_bgr, detected_plates

def run_val_pipeline():
    """Duyệt tập val và lưu kết quả."""
    image_files = [f for f in os.listdir(VAL_IMAGES_PATH) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Đang xử lý {len(image_files)} ảnh từ {VAL_IMAGES_PATH}...")

    for i, file_name in enumerate(image_files):
        img_path = os.path.join(VAL_IMAGES_PATH, file_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        processed_img, plates = core_pipeline(img)
        print(f"[{i+1}] {file_name}: {', '.join(plates) if plates else 'No plate'}")
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"res_{file_name}"), processed_img)

    print(f"\nHoàn tất! Kết quả tại: {OUTPUT_DIR}")

# --- XỬ LÝ VIDEO ---
ocr_history = deque(maxlen=10) 

def get_stable_plate(new_text):
    if new_text:
        ocr_history.append(new_text)
    
    if len(ocr_history) < 5:
        return "" 

    counts = Counter(ocr_history)
    most_common_text, frequency = counts.most_common(1)[0]
    
    if frequency >= 5:
        return most_common_text
    return ""

def process_video(video_path, output_path='output_video.mp4'):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("Đang xử lý video với bộ lọc ổn định 5-frame...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=0.5, verbose=False)[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]
            processed = improve_ocr_input(plate_crop)
            
            # CẬP NHẬT: Thêm dấu '-' và '.' vào allowlist và gọi format_vietnamese_plate
            ocr_res = reader.readtext(processed, allowlist='0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-.')
            current_plate_text = format_vietnamese_plate(ocr_res)
            
            stable_text = get_stable_plate(current_plate_text)
            
            if stable_text:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"STABLE: {stable_text}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('ALPR Stable Filter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # run_val_pipeline()
    process_video('video_test.mp4')