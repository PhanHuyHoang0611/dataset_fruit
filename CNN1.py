import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Định nghĩa mô hình SimpleCNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=102):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Kích thước đầu vào cho lớp Linear đầu tiên sau 3 lần MaxPool2d từ 128x128
        # là 128 kênh * 16x16 pixel
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 2. Chuẩn bị dữ liệu (ĐIỀU CHỈNH KÍCH THƯỚC ẢNH ĐẦU VÀO) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)), # ĐÃ ĐIỀU CHỈNH TẠY ĐÂY
    transforms.ToTensor()
])

# Đảm bảo đường dẫn chính xác đến dataset của bạn
train_dataset = datasets.ImageFolder(r"C:\Users\admin\Dropbox\PC\Desktop\Hocmay\Phanloaihoa\flower_data\train", transform=transform)
val_dataset = datasets.ImageFolder(r"C:\Users\admin\Dropbox\PC\Desktop\Hocmay\Phanloaihoa\flower_data\valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

# --- 3. Khởi tạo mô hình, hàm mất mát, bộ tối ưu hóa ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_cnn = SimpleCNN(num_classes=num_classes).to(device)
criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)

# --- 4. Huấn luyện mô hình SimpleCNN ---
print("\n--- Training SimpleCNN Model ---")
num_epochs_cnn = 50
train_losses_cnn = [] 
val_accuracies_cnn = [] 
for epoch in range(num_epochs_cnn):
    model_cnn.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_cnn.zero_grad()
        outputs = model_cnn(inputs)
        loss = criterion_cnn(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(train_loader)
    train_losses_cnn.append(avg_train_loss)

    model_cnn.eval() 
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad(): 
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_cnn(inputs)
            _, predicted = torch.max(outputs.data, 1) 
            total_samples += labels.size(0) 
            correct_predictions += (predicted == labels).sum().item() 
            
    epoch_accuracy = correct_predictions / total_samples
    val_accuracies_cnn.append(epoch_accuracy)
        
    print(f"Epoch {epoch+1}/{num_epochs_cnn}, Train Loss: {avg_train_loss:.4f}, Val Accuracy: {epoch_accuracy:.4f}")

# --- 5. Đánh giá mô hình SimpleCNN ---
print("\n--- Evaluating SimpleCNN Model ---")
model_cnn.eval()
y_true_cnn, y_pred_cnn = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model_cnn(inputs)
        preds = torch.argmax(outputs, 1).cpu().numpy()
        y_pred_cnn.extend(preds)
        y_true_cnn.extend(labels.numpy())

# In báo cáo phân loại và độ chính xác
print("\nClassification Report (SimpleCNN):")
print(classification_report(y_true_cnn, y_pred_cnn, zero_division=0))

accuracy_cnn = accuracy_score(y_true_cnn, y_pred_cnn)
print(f"Accuracy (SimpleCNN): {accuracy_cnn:.4f}")

# --- 6. VẼ CÁC BIỂU ĐỒ CHO SIMPLECNN ---

# Đồ thị Learning Curves (Loss & Accuracy)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1) # Vị trí 1 trong lưới 1 hàng 2 cột
plt.plot(range(1, num_epochs_cnn + 1), train_losses_cnn, label='Train Loss', color='blue')
plt.title('Figure 2.3: Training Loss per Epoch (SimpleCNN)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2) # Vị trí 2 trong lưới 1 hàng 2 cột
plt.plot(range(1, num_epochs_cnn + 1), val_accuracies_cnn, label='Validation Accuracy', color='orange')
plt.title('Figure 2.4: Validation Accuracy per Epoch (SimpleCNN)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout() # Điều chỉnh layout để tránh chồng lấn
plt.suptitle('Figure 2.2: Learning Curves for SimpleCNN Model', y=1.02) # Tiêu đề chung cho cả 2 đồ thị con
plt.show()


# Vẽ ma trận nhầm lẫn
cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)

plt.figure(figsize=(30, 30)) # Tăng kích thước để dễ nhìn hơn
sns.heatmap(cm_cnn, annot=False, fmt='d', cmap='Blues', cbar=True) # tắt annot để tránh tràn số
plt.title('Confusion Matrix for SimpleCNN Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# --- Sơ đồ mạng SimpleCNN (Mô tả) ---
print("\n--- Kiến trúc mô hình SimpleCNN ---")
print(model_cnn)