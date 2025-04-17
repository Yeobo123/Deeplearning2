import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Kiểm tra xem có GPU không
device = "cuda" if torch.cuda.is_available() else "cpu"

# Đọc dữ liệu từ tệp CSV mới
new_csv_file_path = 'new_data.csv'  # Đổi đường dẫn nếu cần
new_data = pd.read_csv(new_csv_file_path)

# Kiểm tra và loại bỏ bất kỳ giá trị thiếu nào
new_data = new_data.dropna()

# Mã hóa các cột phân loại
new_data_encoded = pd.get_dummies(new_data, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)

# Mã hóa cột 'Drug' thành giá trị số
new_data_encoded['Drug'] = new_data_encoded['Drug'].astype('category').cat.codes

# Chuẩn hóa các cột 'Age' và 'Na_to_K'
scaler = StandardScaler()
new_data_encoded[['Age', 'Na_to_K']] = scaler.fit_transform(new_data_encoded[['Age', 'Na_to_K']])

# Chia dữ liệu thành các đặc trưng và nhãn
X_new = new_data_encoded.drop('Drug', axis=1).values.astype(np.float32)
y_new = new_data_encoded['Drug'].values

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Chuyển đổi thành torch tensors
X_train_tensor_new = torch.tensor(X_train_new, dtype=torch.float32).to(device)
y_train_tensor_new = torch.tensor(y_train_new, dtype=torch.long).to(device)
X_val_tensor_new = torch.tensor(X_val_new, dtype=torch.float32).to(device)
y_val_tensor_new = torch.tensor(y_val_new, dtype=torch.long).to(device)

# Tạo Dataset cho dữ liệu mới
class DrugDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

# Tạo DataLoader cho huấn luyện và kiểm tra
train_data_new = DrugDataset(X_train_tensor_new, y_train_tensor_new)
val_data_new = DrugDataset(X_val_tensor_new, y_val_tensor_new)

train_loader_new = DataLoader(train_data_new, batch_size=32, shuffle=True)
val_loader_new = DataLoader(val_data_new, batch_size=32, shuffle=False)

# Xây dựng mô hình
def get_model():
    model = nn.Sequential(
        nn.Linear(X_train_tensor_new.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.5),  # Tăng tỷ lệ dropout
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),  # Tăng tỷ lệ dropout cho lớp ẩn
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, len(new_data['Drug'].unique()))  # Số lớp đầu ra phụ thuộc vào số lớp trong 'Drug'
    ).to(device)
    return model

# Tải lại mô hình đã huấn luyện (nếu có)
model = get_model()
model.load_state_dict(torch.load('drug_model.pth', weights_only=True))  # Nếu không có mô hình đã huấn luyện, bạn có thể huấn luyện lại từ đầu
model.to(device)

# Định nghĩa hàm huấn luyện cho một batch
def train_batch(x, y, model, optimizer, loss_fn):
    x, y = x.to(device), y.to(device)  # Đảm bảo dữ liệu trên đúng thiết bị
    model.train()
    optimizer.zero_grad()
    prediction = model(x)  # Dự đoán
    batch_loss = loss_fn(prediction, y)  # Tính loss
    batch_loss.backward()  # Tính gradient
    optimizer.step()  # Cập nhật trọng số
    return batch_loss.item()

# Định nghĩa hàm tính độ chính xác
def accuracy(x, y, model):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        _, predicted_labels = torch.max(prediction, 1)
        correct = (predicted_labels == y).sum().item()
        acc = correct / y.size(0)
        acc_percentage = round(acc * 100)  # Nhân với 100 để có phần trăm và làm tròn
        return acc_percentage

# Hàm đánh giá mô hình trên tập kiểm tra
def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)  # Đảm bảo dữ liệu trên đúng thiết bị
            prediction = model(x)
            _, predicted_labels = torch.max(prediction, 1)
            y_true.extend(y.cpu().numpy())  # Lưu nhãn thực tế
            y_pred.extend(predicted_labels.cpu().numpy())  # Lưu nhãn dự đoán
    
    # In báo cáo đánh giá
    print(classification_report(y_true, y_pred, zero_division=0))


# Định nghĩa loss function và optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Huấn luyện mô hình trên dữ liệu mới
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    for x_batch, y_batch in train_loader_new:
        loss = train_batch(x_batch, y_batch, model, optimizer, loss_fn)
        epoch_losses.append(loss)
        epoch_accuracies.append(accuracy(x_batch, y_batch, model))
    
    print(f"Epoch {epoch+1}, Loss: {np.mean(epoch_losses):.4f}, Accuracy: {np.mean(epoch_accuracies):.4f}")
    


# Đánh giá mô hình
evaluate_model(model, val_loader_new)


