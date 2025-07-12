import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    print("✅ CUDA verfügbar!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Anzahl CUDA-Geräte: {torch.cuda.device_count()}")
else:
    print("❌ CUDA nicht verfügbar.")

# 1. GPU/CPU automatisch wählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Verwende Gerät: {device}")

# 2. Hyperparameter
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)

# 3. Daten vorbereiten
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
val_dataset = datasets.ImageFolder(root="dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Modell: MobileNetV2 als CNN
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)  # Output anpassen
model = model.to(device)

EPOCHS = 10 #20  # z. B. von 10 auf 30 erhöhen
LEARNING_RATE = 0.005  # vorher 0.001 → aggressiveres Lernen

# 5. Training vorbereiten
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Optional: dynamischer Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)


# 6. Training starten
best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100
    print(f"🔁 Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total * 100
    print(f" Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_cnn_model.pth")
        print(f" Neues bestes Modell gespeichert bei {val_acc:.2f}%")

print(" Training abgeschlossen.")
