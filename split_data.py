import os
import shutil
import random

# --- CẤU HÌNH ---
DATA_DIR = 'data_images'
IMAGES_DIR = os.path.join(DATA_DIR, 'test')         # Thư mục chứa file .jpg
LABELS_DIR = os.path.join(DATA_DIR, 'test_labels')  # Thư mục chứa file .txt vừa tạo

# Thư mục đầu ra cho YOLO
OUTPUT_ROOT = 'datasets'
TRAIN_PATH = os.path.join(OUTPUT_ROOT, 'train')
VAL_PATH = os.path.join(OUTPUT_ROOT, 'val')

# Tỉ lệ chia (80% train, 20% val)
SPLIT_RATIO = 0.8

def create_folders():
    for path in [TRAIN_PATH, VAL_PATH]:
        os.makedirs(os.path.join(path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(path, 'labels'), exist_ok=True)

def split_dataset():
    create_folders()

    # Lấy danh sách tất cả các file ảnh (không lấy đường dẫn đầy đủ)
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Trộn ngẫu nhiên để đảm bảo tính khách quan
    random.seed(42) # Cố định seed để kết quả giống nhau mỗi lần chạy
    random.shuffle(all_images)

    split_index = int(len(all_images) * SPLIT_RATIO)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    def copy_files(files, target_root):
        for filename in files:
            # Đường dẫn file nguồn
            img_src = os.path.join(IMAGES_DIR, filename)
            label_name = os.path.splitext(filename)[0] + '.txt'
            label_src = os.path.join(LABELS_DIR, label_name)

            if os.path.exists(label_src):
                # Copy ảnh
                shutil.copy(img_src, os.path.join(target_root, 'images', filename))
                # Copy nhãn
                shutil.copy(label_src, os.path.join(target_root, 'labels', label_name))
            else:
                print(f"Bỏ qua {filename} vì không tìm thấy file nhãn tương ứng.")

    print(f"Đang copy {len(train_images)} ảnh vào tập TRAIN...")
    copy_files(train_images, TRAIN_PATH)
    
    print(f"Đang copy {len(val_images)} ảnh vào tập VAL...")
    copy_files(val_images, VAL_PATH)

    print("\n--- HOÀN TẤT ---")
    print(f"Dữ liệu đã sẵn sàng tại thư mục: {OUTPUT_ROOT}")

if __name__ == "__main__":
    split_dataset()