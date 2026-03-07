from ultralytics import YOLO

def train_model():
    # Tiếp tục dùng bản Nano vì nó hoàn hảo cho VRAM 6GB và chạy realtime rất mượt
    model = YOLO('yolov8n.pt')

    print("🚀 Bắt đầu quá trình huấn luyện trên GPU (6GB VRAM)...")

    results = model.train(
        data='data.yaml',      
        epochs=100,            
        imgsz=640,             
        batch=8,               # THAY ĐỔI QUAN TRỌNG: Hạ xuống 8 là mức an toàn tuyệt đối để không bị tràn VRAM.
        patience=20,           
        device=0,              
        workers=4,             # THAY ĐỔI QUAN TRỌNG: Giảm xuống 4 để phù hợp với 16GB RAM hệ thống, tránh thắt cổ chai CPU.
        amp=True,              # VŨ KHÍ BẢO VỆ VRAM: Bắt buộc giữ True để giảm một nửa dung lượng nhớ GPU cần dùng.
        name='alpr_v1_6gb_vram',   
        exist_ok=True,         
        pretrained=True,       
        optimizer='auto',      
        lr0=0.01,              
        cos_lr=True            
    )

    print("\n--- 🎉 HUẤN LUYỆN HOÀN TẤT ---")
    print("Mô hình tốt nhất nằm tại: runs/detect/alpr_v1_6gb_vram/weights/best.pt")

if __name__ == "__main__":
    train_model()