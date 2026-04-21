import os

def rename_to_jake(folder_path):
    print("Đang xử lý đổi tên thành Jake, vui lòng đợi...")
    
    # BƯỚC 1: Lọc file và đổi đuôi tạm (.tmp) để tránh xung đột
    temp_files = []
    for filename in os.listdir(folder_path):
        # Lệnh này đảm bảo chỉ nhắm mục tiêu vào các file của Speaker_0000
        if filename.startswith("Speaker_0001") and filename.endswith(".wav"):
            old_path = os.path.join(folder_path, filename)
            temp_path = old_path + ".tmp"
            os.rename(old_path, temp_path)
            temp_files.append(temp_path)
            
    # BƯỚC 2: Đánh số thứ tự lại từ 1 với tên "Jake"
    count = 1
    for temp_path in temp_files:
        # Tên mới sẽ cực kỳ chuẩn mực: Jake_1.wav, Jake_2.wav...
        new_filename = f"Will_{count}.wav"
        new_path = os.path.join(folder_path, new_filename)
        
        # Đổi lại tên chính thức và xóa đuôi .tmp
        os.rename(temp_path, new_path)
        count += 1
        
    print(f"🎉 Hoàn tất! Đã đổi tên thành công {count - 1} file thành định dạng 'Jake_số.wav'")

# ==========================================
# ĐƯỜNG DẪN TỚI THƯ MỤC CHỨA FILE CẦN ĐỔI
# ==========================================
THU_MUC_VOICES = r"C:\Users\ducno\OneDrive\Desktop\Speaker_0001"

rename_to_jake(THU_MUC_VOICES)