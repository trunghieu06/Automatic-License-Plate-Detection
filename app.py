import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import easyocr
import re
from collections import deque, Counter

# --- 1. CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(page_title="ALPR System", page_icon="🚗", layout="wide")
st.title("🚗 Hệ thống Nhận diện Biển số Xe Việt Nam")
st.markdown("Hệ thống tích hợp YOLOv8, EasyOCR và Heuristic Rules để nhận diện chính xác 2 dòng.")

# --- 2. TỐI ƯU HÓA: NẠP MÔ HÌNH VÀO CACHE ---
# @st.cache_resource giúp Streamlit chỉ nạp mô hình 1 lần duy nhất khi bật web
@st.cache_resource
def load_models():
    # Đảm bảo đường dẫn này đúng với máy của bạn
    yolo_model = YOLO('best.pt')
    ocr_reader = easyocr.Reader(['en'])
    return yolo_model, ocr_reader

model, reader = load_models()

# --- 3. BỘ TỪ ĐIỂN VÀ LUẬT HẬU XỬ LÝ ---
CHAR_TO_NUM = {'O': '0', 'D': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}

def force_number(text):
    return ''.join([CHAR_TO_NUM.get(c, c) for c in text])

def format_vietnamese_plate(ocr_results):
    if not ocr_results: return ""
    sorted_lines = sorted(ocr_results, key=lambda x: x[0][0][1])
    if len(sorted_lines) < 2: return sorted_lines[0][1].upper()

    top_line_raw = sorted_lines[0][1].upper()
    bottom_line_raw = sorted_lines[1][1].upper()

    top_clean = re.sub(r'[^A-Z0-9]', '', top_line_raw)
    if len(top_clean) >= 4:
        province_code = force_number(top_clean[:2])
        seri_code = top_clean[2:4]
        top_final = f"{province_code}-{seri_code}"
    else:
        top_final = top_clean 

    bottom_clean = re.sub(r'[^A-Z0-9]', '', bottom_line_raw)
    bottom_nums = force_number(bottom_clean)
    if len(bottom_nums) == 5:
        bottom_final = f"{bottom_nums[:3]}.{bottom_nums[3:]}"
    elif len(bottom_nums) == 4:
        bottom_final = bottom_nums
    else:
        bottom_final = bottom_nums 

    return f"{top_final} {bottom_final}"

def improve_ocr_input(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray_enlarged = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray_enlarged, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    final_thresh = cv2.bitwise_not(thresh)
    return final_thresh

def core_pipeline(image_bgr):
    results = model(image_bgr, conf=0.5, verbose=False)[0]
    detected_plates = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        
        plate_crop = image_bgr[y1:y2, x1:x2]
        if plate_crop.size == 0: continue
            
        processed_plate = improve_ocr_input(plate_crop)
        ocr_results = reader.readtext(processed_plate, allowlist='0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-.')
        text = format_vietnamese_plate(ocr_results)
        
        detected_plates.append(text)
        
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{text} ({conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image_bgr, (x1, y1 - h - 15), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
    return image_bgr, detected_plates

# --- LƯU TRỮ LỊCH SỬ CHO VIDEO ---
ocr_history = deque(maxlen=10) 
def get_stable_plate(new_text):
    if new_text: ocr_history.append(new_text)
    if len(ocr_history) < 5: return "" 
    counts = Counter(ocr_history)
    most_common_text, frequency = counts.most_common(1)[0]
    if frequency >= 5: return most_common_text
    return ""

# --- 4. GIAO DIỆN NGƯỜI DÙNG TƯƠNG TÁC ---
app_mode = st.sidebar.selectbox("Chọn chế độ hoạt động:", ["Tải ảnh lên (Upload)", "Camera trực tiếp (Live)"])

if app_mode == "Tải ảnh lên (Upload)":
    st.header("🖼️ Nhận diện qua Hình ảnh")
    uploaded_file = st.file_uploader("Tải lên ảnh chứa biển số xe", type=["jpg", "jpeg", "png"])
    

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        image = ImageOps.exif_transpose(image) 
        
        img_array = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("Đang chạy mô hình AI..."):
            processed_img, plates = core_pipeline(img_bgr)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc", use_container_width=True)
        with col2:
            res_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="Kết quả nhận diện", use_container_width=True)

        if plates:
            st.success(f"**Kết quả:** Đã tìm thấy {len(plates)} biển số: {', '.join(plates)}")
        else:
            st.warning("Không tìm thấy biển số nào trong ảnh.")

elif app_mode == "Camera trực tiếp (Live)":
    st.header("🎥 Nhận diện qua Camera (Real-time)")
    st.markdown("Đưa biển số vào khung hình camera để hệ thống nhận diện.")
    
    from streamlit_webrtc import webrtc_streamer
    import av

    # Hàm xử lý từng frame video truyền từ điện thoại/webcam lên
    def video_frame_callback(frame):
        # Lấy frame ảnh
        img = frame.to_ndarray(format="bgr24")
        
        # Đưa qua Pipeline nhận diện biển số của bạn
        processed_img, _ = core_pipeline(img)
        
        # Trả ảnh đã vẽ khung về lại màn hình
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    # Khởi chạy giao diện Camera WebRTC
    # Khởi chạy giao diện Camera WebRTC
    webrtc_streamer(
        key="alpr-camera",
        video_frame_callback=video_frame_callback,
        rtc_configuration={  
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        # SỬA LỖI 1: Ép điện thoại mở Camera sau (environment) thay vì Selfie (user)
        media_stream_constraints={
            "video": {"facingMode": "environment"}, 
            "audio": False
        },
        # SỬA LỖI 2: Fix lỗi dải hẹp, ép khung video mở rộng 100% màn hình
        video_html_attrs={
            "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #00FF00"},
            "controls": False,
            "autoPlay": True,
        }
    )