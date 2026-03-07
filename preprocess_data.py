import cv2
import os

# --- CẤU HÌNH ---
LOCATION_FILE = 'data_images/location.txt'
IMAGES_DIR = 'data_images/test'  # Thư mục chứa các file .jpg thực tế
OUTPUT_DIR = 'data_images/test_labels' # Nơi lưu các file .txt mới

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def convert_to_yolo():
    with open(LOCATION_FILE, 'r') as f:
        lines = f.readlines()

    print(f"Bắt đầu xử lý {len(lines)} dòng dữ liệu...")

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6: continue

        img_name = parts[0]      # Ví dụ: 0000_00532_b.jpg
        class_id = int(parts[1]) - 1  # YOLO thường bắt đầu từ 0 (nếu file của bạn 1 là biển số)
        xmin = float(parts[2])
        ymin = float(parts[3])
        w_box = float(parts[4])
        h_box = float(parts[5])

        # 1. Đọc ảnh để lấy kích thước thực tế (W, H)
        img_path = os.path.join(IMAGES_DIR, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Cảnh báo: Không tìm thấy ảnh {img_name}, bỏ qua dòng này.")
            continue
        
        h_img, w_img, _ = img.shape

        # 2. Tính toán Center X, Center Y và Chuẩn hóa sang [0, 1]
        # Công thức YOLO: x_center, y_center, width, height (tất cả / W_img hoặc H_img)
        x_center = (xmin + w_box / 2.0) / w_img
        y_center = (ymin + h_box / 2.0) / h_img
        w_norm = w_box / w_img
        h_norm = h_box / h_img

        # 3. Tạo file .txt tương ứng (ví dụ: 0000_00532_b.txt)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(OUTPUT_DIR, txt_name)

        with open(txt_path, 'w') as f_out:
            # Ghi theo định dạng: class x_center y_center width height
            line_to_write = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            f_out.write(line_to_write)

    print(f"Hoàn tất! Các file nhãn đã được lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_to_yolo()