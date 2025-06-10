import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Định nghĩa mô hình ResNet18 TỰ XÂY (từ đầu) ---

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=102):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- 2. Chuẩn bị dữ liệu ---
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Kích thước ảnh đầu vào 224x224 cho ResNet18
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(r"C:\Users\admin\Dropbox\PC\Desktop\Hocmay\Phanloaihoa\flower_data\train", transform=transform)
val_dataset = datasets.ImageFolder(r"C:\Users\admin\Dropbox\PC\Desktop\Hocmay\Phanloaihoa\flower_data\valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

# --- 3. Khởi tạo mô hình, hàm mất mát, bộ tối ưu hóa ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_resnet = ResNet18(num_classes=num_classes).to(device)
criterion_resnet = nn.CrossEntropyLoss()
optimizer_resnet = optim.Adam(model_resnet.parameters(), lr=0.001) # Tối ưu toàn bộ tham số

# --- 4. Huấn luyện và Đánh giá mô hình ResNet18 theo từng Epoch ---
print("\n--- Training ResNet18 (Custom) Model with Accuracy per Epoch ---")
num_epochs_resnet = 50
train_losses_resnet = [] 
val_accuracies_resnet = [] 
for epoch in range(num_epochs_resnet):
    model_resnet.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_resnet.zero_grad()
        outputs = model_resnet(inputs)
        loss = criterion_resnet(outputs, labels)
        loss.backward()
        optimizer_resnet.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses_resnet.append(avg_train_loss)

    model_resnet.eval()
    y_true_val, y_pred_val = [], []
    with torch.no_grad():
        correct_predictions = 0
        total_samples = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_resnet(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            y_pred_val.extend(predicted.cpu().numpy())
            y_true_val.extend(labels.cpu().numpy())
            
    epoch_accuracy = correct_predictions / total_samples
    val_accuracies_resnet.append(epoch_accuracy)
    print(f"Epoch {epoch+1}/{num_epochs_resnet}, Train Loss: {avg_train_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")

# --- 5. Đánh giá cuối cùng của mô hình ResNet18 ---
print("\n--- Final Evaluation of Custom ResNet18 Model ---")
print("\nClassification Report (ResNet18):")
print(classification_report(y_true_val, y_pred_val, zero_division=0))

accuracy_resnet = accuracy_score(y_true_val, y_pred_val)
print(f"Final Accuracy (ResNet18): {accuracy_resnet:.4f}")

# Đồ thị Learning Curves (Loss & Accuracy)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1) # Vị trí 1 trong lưới 1 hàng 2 cột
plt.plot(range(1, num_epochs_resnet + 1), train_losses_resnet, label='Train Loss', color='blue')
plt.title('Figure 2.3: Training Loss per Epoch (Custom ResNet18)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2) # Vị trí 2 trong lưới 1 hàng 2 cột
plt.plot(range(1, num_epochs_resnet + 1), val_accuracies_resnet, label='Validation Accuracy', color='orange')
plt.title('Figure 2.4: Validation Accuracy per Epoch (Custom ResNet18)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout() # Điều chỉnh layout để tránh chồng lấn
plt.suptitle('Figure 2.2: Learning Curves for Custom ResNet18 Model', y=1.02) # Tiêu đề chung cho cả 2 đồ thị con
plt.show()
# Vẽ ma trận nhầm lẫn
cm_resnet = confusion_matrix(y_true_val, y_pred_val)
plt.figure(figsize=(30, 30))
sns.heatmap(cm_resnet, annot=False, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix for Custom ResNet18 Model (224x224 input)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\n--- Kiến trúc mô hình ResNet18 (Custom Build) ---")
print(model_resnet)