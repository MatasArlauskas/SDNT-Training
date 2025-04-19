# Download dataset
!wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

import tarfile
import os
import scipy.io
from collections import Counter
import numpy as np

# Išskleidžiame archyvą
with tarfile.open("102flowers.tgz", "r:gz") as tar:
    tar.extractall(path="flowers")

# Atsisiunčiame etikečių failą
!wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

# Įkeliame etiketes
labels_data = scipy.io.loadmat("imagelabels.mat")
labels = labels_data["labels"][0]  # 1-indexed

# Suskaičiuojame kiekvienos klasės dydį
label_counts = Counter(labels)

# Pasirenkame top 5 klases su daugiausiai pavyzdžių
top_classes = [label for label, _ in label_counts.most_common(5)]
print("Pasirinktos klasės:", top_classes)

# Filtruojame tik atitinkamus paveikslėlius
image_dir = "flowers/jpg/"
all_images = sorted(os.listdir(image_dir))  # turi sutapti su etikečių eiliškumu

# Sukuriame žemėlapį: pvz. {51: 0, 77: 1, 46: 2, 73: 3, 89: 4}
label_mapping = {label: idx for idx, label in enumerate(top_classes)}

X = []
y = []

for i, image_name in enumerate(all_images):
    original_label = labels[i]
    if original_label in top_classes:
        X.append(os.path.join(image_dir, image_name))
        y.append(label_mapping[original_label])  # čia jau nuo 0 iki 4

print("Unikalūs y:", set(y))  # ← TURĖTŲ parodyti: {0, 1, 2, 3, 4}
print(f"Iš viso atrinkta paveikslėlių: {len(X)}")


from sklearn.model_selection import train_test_split

# Pirmiausia – atskiriame 20% (bus skirta validacijai ir testavimui)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Dabar padaliname tą 20% į dvi lygias dalis: 10% test ir 10% val
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Mokymo vaizdų: {len(X_train)}")
print(f"Validacijos vaizdų: {len(X_val)}")
print(f"Testavimo vaizdų: {len(X_test)}")


import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Transformacija: dydžio keitimas, normalizavimas
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Skalė [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # → [-1, 1]
])

class FlowerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx] 
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataset'ai
train_dataset = FlowerDataset(X_train, y_train, transform=transform)
val_dataset = FlowerDataset(X_val, y_val, transform=transform)
test_dataset = FlowerDataset(X_test, y_test, transform=transform)

# DataLoader'ai
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Apkarpymas ir keitimas
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Sukimas, poslinkis, zoom
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Ryškumas, spalvų drebėjimas
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Skalė [-1, 1]
])




# Tik treniravimo dataset'ui – su augmentacija
train_dataset = FlowerDataset(X_train, y_train, transform=train_transform)

# Validacijai ir testavimui – be augmentacijos, tik normalizacija
val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_dataset = FlowerDataset(X_val, y_val, transform=val_test_transform)
test_dataset = FlowerDataset(X_test, y_test, transform=val_test_transform)




from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# SDNT struktūros paruošimas
import torch
import torch.nn as nn
import torch.nn.functional as F

class SDNT(nn.Module):
    def __init__(self, num_classes):
        super(SDNT, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),  # 64x64

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),  # 32x32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),  # 16x16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # (128, 16, 16) → 128*16*16
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




num_classes = 5  # 5 klases
model = SDNT(num_classes).to("cuda" if torch.cuda.is_available() else "cpu")




print("Unikalūs y_train:", set(y_train))
print("Unikalūs y_val:", set(y_val))
print("Unikalūs y_test:", set(y_test))



# Training
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicijuojame modelį, loss ir optimizatorių
model = SDNT(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())

num_epochs = 15  # arba galima ir daugiau

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Išsaugome geriausią modelį
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ Saved best model")

# Išsaugome paskutinės epochos modelį
torch.save(model.state_dict(), "last_model.pt")
print("✅ Saved last model")



import matplotlib.pyplot as plt

# Loss kreivės
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Mokymo ir Validacijos Loss")
plt.legend()
plt.grid(True)
plt.show()

# Accuracy kreivės
plt.figure(figsize=(10, 4))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Mokymo ir Validacijos Tikslumas")
plt.legend()
plt.grid(True)
plt.show()



# Įkeliame geriausią modelį
model.load_state_dict(torch.load("best_model.pt"))
model.eval()



from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Painiavos matrica
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Spėta klasė")
plt.ylabel("Tikra klasė")
plt.title("Painiavos matrica")
plt.show()

# Klasifikavimo ataskaita
print("📋 Klasifikavimo ataskaita:")
print(classification_report(all_labels, all_preds, digits=4))



import random
import matplotlib.pyplot as plt

class_names = [f"Klase {i}" for i in range(5)]  

def show_predictions(model, dataset, n=8):
    model.eval()
    indices = random.sample(range(len(dataset)), n)
    images_labels = [dataset[i] for i in indices]

    plt.figure(figsize=(15, 4))
    for i, (image, true_label) in enumerate(images_labels):
        image_input = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_input)
            _, pred_label = torch.max(output, 1)
        image_np = image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # Atstatom normalizavimą

        plt.subplot(1, n, i + 1)
        plt.imshow(image_np)
        plt.axis("off")
        plt.title(f"Tikra: {class_names[true_label]}\nSpėta: {class_names[pred_label.item()]}")
    plt.tight_layout()
    plt.show()

# Rodome 8 pavyzdžius
show_predictions(model, test_dataset, n=8)



