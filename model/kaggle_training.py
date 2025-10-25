import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Kiểm tra cấu trúc thư mục
data_dir = Path('/kaggle/input/hagrid-sample/other/default/1/hagrid-sample-30k-384p')
hagrid_dir = data_dir / 'hagrid_30k'
print("Nội dung thư mục data_dir:", os.listdir(data_dir))
print("Nội dung thư mục hagrid_30k:", os.listdir(hagrid_dir))

# Định nghĩa các lớp cử chỉ (loại bỏ 'like' và 'three2' vì không có ảnh)
classes = [
    'call', 'dislike', 'fist', 'four', 'mute', 'ok', 'one', 'palm',
    'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three',
    'two_up', 'two_up_inverted'
]
num_classes = len(classes)
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# Định nghĩa dataset
class HaGRIDDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load tất cả ảnh từ các thư mục train_val_{class}
all_images = []
all_labels = []
for cls in classes:
    cls_path = hagrid_dir / f'train_val_{cls}'
    if cls_path.exists():
        images = [str(p) for p in cls_path.glob('*.jpg')]
        all_images.extend(images)
        all_labels.extend([class_to_idx[cls]] * len(images))
    else:
        print(f"Thư mục không tồn tại: {cls_path}")

if not all_images:
    raise ValueError("Không tìm thấy hình ảnh nào trong dataset!")

# Chia tập train/test (80/20)
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# Transform dữ liệu
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Tạo dataset
train_dataset = HaGRIDDataset(train_imgs, train_labels, transform=data_transforms['train'])
val_dataset = HaGRIDDataset(val_imgs, val_labels, transform=data_transforms['val'])
print(f"Số mẫu train: {len(train_dataset)}")
print(f"Số mẫu val: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Khởi tạo mô hình ResNet-50
model = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Lưu lịch sử huấn luyện
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0.0
best_model_path = '/kaggle/working/best_model.pth'

# Hàm trích xuất đặc trưng cho t-SNE
def get_features(model, loader, layer='fc'):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            if layer == 'penultimate':
                x = model.conv1(inputs)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                features.append(x.cpu().numpy())
            else:
                outputs = model(inputs)
                features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

# Huấn luyện mô hình
num_epochs = 20
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    # Đánh giá trên tập validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Lưu mô hình tốt nhất
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model with Val Acc: {best_val_acc:.4f}')
    
    scheduler.step()

training_time = time.time() - start_time
print(f'Total Training Time: {training_time/60:.2f} minutes')

# Vẽ biểu đồ loss và accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('/kaggle/working/loss_accuracy_plot.png')
plt.show()

# Tính toán các chỉ số đánh giá
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
accuracy = accuracy_score(all_labels, all_preds)
print(f'F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

# Vẽ ROC và tính AUC
y_true_bin = label_binarize(all_labels, classes=range(num_classes))
y_score = []
model.eval()
with torch.no_grad():
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        y_score.append(torch.softmax(outputs, dim=1).cpu().numpy())
y_score = np.concatenate(y_score)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {classes[i]}) (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='best')
plt.savefig('/kaggle/working/roc_curves.png')
plt.show()

# t-SNE cho layer cuối và layer gần cuối
features_final, labels_final = get_features(model, val_loader, layer='fc')
features_penultimate, _ = get_features(model, val_loader, layer='penultimate')

tsne = TSNE(n_components=2, random_state=42)
tsne_results_final = tsne.fit_transform(features_final)
tsne_results_penultimate = tsne.fit_transform(features_penultimate)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(num_classes):
    idx = labels_final == i
    plt.scatter(tsne_results_final[idx, 0], tsne_results_final[idx, 1], label=classes[i], alpha=0.5)
plt.title('t-SNE of Final Layer')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(num_classes):
    idx = labels_final == i
    plt.scatter(tsne_results_penultimate[idx, 0], tsne_results_penultimate[idx, 1], label=classes[i], alpha=0.5)
plt.title('t-SNE of Penultimate Layer')
plt.legend()
plt.savefig('/kaggle/working/tsne_plots.png')
plt.show()