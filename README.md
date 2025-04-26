# 💊 Dự án Phân Loại Thuốc bằng Deep Learning với PyTorch

Dự án này sử dụng Deep Learning (PyTorch) để dự đoán loại thuốc phù hợp cho từng bệnh nhân dựa trên các đặc trưng đầu vào như: độ tuổi, giới tính, huyết áp, cholesterol và tỉ lệ natri-kali trong cơ thể.

✅ Mục tiêu
Xây dựng một mô hình mạng nơ-ron đơn giản để phân loại loại thuốc dựa trên dữ liệu y tế.

Huấn luyện mô hình bằng PyTorch với tập dữ liệu drug200.csv.

Dự đoán loại thuốc phù hợp cho dữ liệu mới (new_data.csv).

Lưu và tải mô hình .pth để phục vụ kiểm thử sau này.

🧠 Mô tả mô hình
Mô hình gồm:

Một neural network nhiều lớp (Multi-Layer Perceptron).

Các lớp Linear, ReLU, Dropout, Softmax.

Sử dụng:

Hàm mất mát: CrossEntropyLoss

Thuật toán tối ưu: Adam

Epoch: bạn có thể tùy chỉnh

Dự đoán đầu ra là một trong các loại thuốc: DrugY, drugA, drugB, drugC, drugX.

📁 Cấu trúc thư mục
bash
Sao chép
Chỉnh sửa
Deeplearning2/
├── chaythu.py          # Huấn luyện mô hình từ dữ liệu

├── test2.py            # Dự đoán thuốc từ dữ liệu mới

├── drug200.csv         # Tập dữ liệu gốc để huấn luyện

├── new_data.csv        # Dữ liệu đầu vào mới để kiểm thử

├── drug_model.pth      # File lưu mô hình đã huấn luyện

├── DEEPLEARNINGG.docx  # Tài liệu báo cáo mô tả chi tiết

└── .gitignore

📊 Dữ liệu
drug200.csv: Gồm 6 cột:

Age (tuổi)

Sex (giới tính)

BP (huyết áp)

Cholesterol (mức cholesterol)

Na_to_K (tỷ lệ natri trên kali)

Drug (loại thuốc đã được kê)

Dữ liệu dạng categorical sẽ được mã hóa (Label Encoding) trước khi đưa vào mô hình.

🚀 Hướng dẫn chạy
1. Cài đặt các thư viện cần thiết
bash
Sao chép
Chỉnh sửa
pip install torch pandas scikit-learn matplotlib
2. Huấn luyện mô hình
bash
Sao chép
Chỉnh sửa
python chaythu.py
Mô hình sẽ:

Tiền xử lý dữ liệu

Huấn luyện mạng nơ-ron

Lưu mô hình vào drug_model.pth

3. Dự đoán dữ liệu mới
bash
Sao chép
Chỉnh sửa
python test2.py
Mã sẽ:

Tải mô hình từ file .pth

Đọc dữ liệu từ new_data.csv

In ra loại thuốc dự đoán cho từng bệnh nhân

📈 Kết quả mong đợi
Độ chính xác mô hình phụ thuộc vào số epoch, kiến trúc mạng và preprocessing.

Bạn có thể thêm biểu đồ loss/accuracy để trực quan hóa quá trình huấn luyện.

📝 Ghi chú
Dự án có thể được mở rộng bằng cách:

Thử nghiệm với các kiến trúc mạng sâu hơn

Dùng các kỹ thuật như BatchNorm, Early Stopping

Đánh giá bằng Precision, Recall, F1-score thay vì chỉ Accuracy
