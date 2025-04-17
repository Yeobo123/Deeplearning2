import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Kiểm tra xem có GPU không
device = "cuda" if torch.cuda.is_available() else "cpu"

# Đọc dữ liệu từ tệp CSV
csv_file_path = 'drug200.csv'
data = pd.read_csv(csv_file_path)

# Kiểm tra và loại bỏ bất kỳ giá trị thiếu nào
data = data.dropna()

# Mã hóa các cột phân loại
data_encoded = pd.get_dummies(data, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)

# Mã hóa cột 'Drug' thành giá trị số
data_encoded['Drug'] = data_encoded['Drug'].astype('category').cat.codes

# Tính toán IQR cho mỗi cột đặc trưng (Age và Na_to_K)
Q1 = data_encoded[['Age', 'Na_to_K']].quantile(0.25)
Q3 = data_encoded[['Age', 'Na_to_K']].quantile(0.75)
IQR = Q3 - Q1

# Loại bỏ các dữ liệu bất thường (outliers) dựa trên IQR
data_cleaned = data_encoded[~((data_encoded[['Age', 'Na_to_K']] < (Q1 - 1.5 * IQR)) | 
                              (data_encoded[['Age', 'Na_to_K']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Chuẩn hóa các cột 'Age' và 'Na_to_K' sau khi xử lý outliers
scaler = StandardScaler()

# Dùng .loc để tránh cảnh báo 'SettingWithCopyWarning'
data_cleaned.loc[:, ['Age', 'Na_to_K']] = scaler.fit_transform(data_cleaned[['Age', 'Na_to_K']]).round().astype('int64')

# Chia dữ liệu thành các đặc trưng và nhãn
X_cleaned = data_cleaned.drop('Drug', axis=1).values.astype(np.float32)
y_cleaned = data_cleaned['Drug'].values

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_val, y_train, y_val = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Chuyển đổi thành torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# Tạo Dataset tùy chỉnh cho dữ liệu huấn luyện và kiểm tra
class DrugDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

# Tạo DataLoader cho huấn luyện và kiểm tra
train_data = DrugDataset(X_train_tensor, y_train_tensor)
val_data = DrugDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Định nghĩa mô hình mạng nơ-ron
def get_model():
    model = nn.Sequential(
        nn.Linear(X_train_tensor.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.5), # Tăng tỷ lệ dropout
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5), # Tăng tỷ lệ dropout cho lớp ẩn
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, len(data['Drug'].unique()))
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Adam optimizer với L2 regularization
    return model, loss_fn, optimizer

# Hàm huấn luyện cho một batch
def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    prediction = model(x) # Dự đoán
    batch_loss = loss_fn(prediction, y) # Tính loss
    batch_loss.backward() # Tính gradient
    optimizer.step() # Cập nhật trọng số
    return batch_loss.item()

# Hàm tính độ chính xác
def accuracy(x, y, model):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        _, predicted_labels = torch.max(prediction, 1)
        correct = (predicted_labels == y).sum().item()
        return correct / y.size(0)

# Hàm đánh giá mô hình trên tập kiểm tra
def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x, y in val_loader:
            prediction = model(x)
            _, predicted_labels = torch.max(prediction, 1)
            y_true.extend(y.cpu().numpy()) # Lưu nhãn thực tế
            y_pred.extend(predicted_labels.cpu().numpy()) # Lưu nhãn dự đoán
    
    # In báo cáo đánh giá
    print(classification_report(y_true, y_pred, zero_division=0))

# Hàm huấn luyện toàn bộ
model, loss_fn, optimizer = get_model()
losses, accuracies = [], []

# Huấn luyện trong 50 epoch
for epoch in range(50): # Tăng số lượng epoch lên 50
    print(f'Epoch {epoch + 1}')
    epoch_losses, epoch_accuracies = [], []
    
    # Huấn luyện trên từng batch
    for x, y in train_loader:
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        epoch_losses.append(batch_loss)
        
        # Tính độ chính xác
        batch_accuracy = accuracy(x, y, model)
        epoch_accuracies.append(batch_accuracy)

    epoch_loss = np.mean(epoch_losses)
    epoch_accuracy = np.mean(epoch_accuracies)
    
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

    # In độ chính xác và loss dưới dạng phần trăm
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy * 100:.2f}%")

# Đánh giá mô hình trên tập kiểm tra
evaluate_model(model, val_loader)

# Vẽ đồ thị Loss và Accuracy
epochs = np.arange(1, 51) # 50 epochs
plt.figure(figsize=(20, 5))

# Đồ thị Loss
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()

# Đồ thị Accuracy
plt.subplot(122)
plt.title('Accuracy value over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
plt.legend()

plt.show()

# Lưu mô hình đã huấn luyện
torch.save(model.state_dict(), 'drug_model.pth')
print("Mô hình đã được lưu.")
