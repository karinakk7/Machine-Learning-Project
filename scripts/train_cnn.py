import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from PIL import Image

class VideoSequenceDataset(Dataset):
    def __init__(self, train_dir, img_size=(224, 224), sequence_length=10, transform=None):
        self.train_dir = train_dir
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Lade alle Sequenzen
        self.sequences, self.frequency_features, self.labels = self.load_video_sequences()
        
        # Label-Mapping erstellen
        self.unique_labels = list(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.labels_encoded = [self.label_to_idx[label] for label in self.labels]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]  # Shape: (sequence_length, H, W, 3)
        frequency_features = self.frequency_features[idx]
        label = self.labels_encoded[idx]
        
        # Konvertiere zu PyTorch Format (sequence_length, 3, H, W)
        sequence_tensor = torch.from_numpy(sequence).permute(0, 3, 1, 2).float()
        frequency_tensor = torch.from_numpy(frequency_features).float()
        
        if self.transform:
            # Transformationen auf jedes Frame anwenden
            transformed_frames = []
            for frame in sequence_tensor:
                transformed_frames.append(self.transform(frame))
            sequence_tensor = torch.stack(transformed_frames)
        
        return sequence_tensor, frequency_tensor, torch.tensor(label, dtype=torch.long)
    
    def load_video_sequences(self, sequence_length=None):
        """
        Lädt Bildsequenzen aus den Trainingsordnern
        
        Args:
            sequence_length: Anzahl aufeinanderfolgender Frames pro Sequenz
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        X_sequences = []
        X_frequency = []
        y_labels = []
        
        print("Lade Video-Sequenzen...")
        
        for class_name in os.listdir(self.train_dir):
            class_path = os.path.join(self.train_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            print(f"Verarbeite Klasse: {class_name}")
            
            # Bilder sortiert laden
            image_files = sorted([f for f in os.listdir(class_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if len(image_files) < sequence_length:
                print(f"Warnung: Klasse {class_name} hat nur {len(image_files)} Bilder, aber {sequence_length} sind erforderlich")
                continue
            
            # Weniger überlappende Sequenzen für bessere Diversität
            step_size = max(1, sequence_length // 4)
            # Sequenzen erstellen
            for i in range(0, len(image_files) - sequence_length * 3 + 1, step_size):
                sequence = []
                
                # Sampling über 3x so viele Frames
                end_idx = min(i + sequence_length * 3 - 1, len(image_files) - 1)
                sampling_indices = np.linspace(i, end_idx, sequence_length, dtype=int)

                for idx in sampling_indices:
                    if idx < len(image_files):
                        img_path = os.path.join(class_path, image_files[idx])
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR zu RGB konvertieren
                            img_resized = cv2.resize(img, self.img_size)
                            # Normalisierung hinzufügen
                            img_normalized = img_resized.astype(np.float32) / 255.0
                            sequence.append(img_normalized)
                
                # Wenn weniger als sequence_length Bilder, mit dem letzten Bild auffüllen
                if len(sequence) > 0:
                    while len(sequence) < sequence_length:
                        sequence.append(sequence[-1])
                    
                    if len(sequence) == sequence_length:
                        # Bildsequenz für CNN/LSTM
                        X_sequences.append(np.array(sequence))
                        
                        # Frequenz-Features extrahieren (für andere Modelle)
                        try:
                            # Zurück zu 0-255 für Frequenz-Features
                            sequence_uint8 = [(seq * 255).astype(np.uint8) for seq in sequence]
                            freq_features = self.extract_frequency_features(sequence_uint8)
                            X_frequency.append(np.mean(freq_features, axis=0))  # Durchschnitt über Sequenz
                        except Exception as e:
                            print(f"Fehler bei Frequenz-Features: {e}")
                            # Fallback: Dummy-Features
                            X_frequency.append(np.zeros(260))  # Standard Feature-Größe
                        
                        y_labels.append(class_name)
        
        print(f"Geladene Sequenzen: {len(X_sequences)}")
        return np.array(X_sequences), np.array(X_frequency), np.array(y_labels)
    
    def extract_frequency_features(self, sequence):
        """
        Placeholder für Frequenz-Feature-Extraktion
        """
        # Hier würdest du deine spezifische Frequenz-Feature-Extraktion implementieren
        features = []
        for frame in sequence:
            # Beispiel: Einfache Statistiken als Features
            feature = np.concatenate([
                frame.mean(axis=(0, 1)),  # Durchschnitt pro Kanal (3 Features)
                frame.std(axis=(0, 1)),   # Standardabweichung pro Kanal (3 Features)
                frame.flatten()[:254]     # Erste 254 Pixel-Werte
            ])
            features.append(feature)
        return np.array(features)


class ImprovedTemporalCNN(nn.Module):
    """
    VERBESSERTE Temporal CNN Architektur
    """
    def __init__(self, num_classes, sequence_length, img_size=(224, 224)):
        super(ImprovedTemporalCNN, self).__init__()
        self.sequence_length = sequence_length
        
        # CNN Feature Extractor (wird auf jedes Frame angewendet)
        self.cnn_features = nn.Sequential(
            # Erste Conv-Block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Zweite Conv-Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Dritte Conv-Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Feature-Reduktion
        self.feature_reduction = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
        )
        
        # Bidirektionale LSTM
        self.lstm1 = nn.LSTM(256, 128, batch_first=True, bidirectional=True, dropout=0.3)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True, dropout=0.3)  # 256 wegen bidirectional
        
        # Klassifikations-Schichten
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),  # 128 wegen bidirectional
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, 3, H, W)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape für CNN: (batch_size * sequence_length, 3, H, W)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        
        # CNN Feature Extraktion
        features = self.cnn_features(x)  # (batch_size * seq_len, 128, 1, 1)
        features = features.view(batch_size * seq_len, -1)  # (batch_size * seq_len, 128)
        
        # Feature-Reduktion
        features = self.feature_reduction(features)  # (batch_size * seq_len, 256)
        
        # Zurück zu Sequenz-Format: (batch_size, sequence_length, 256)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM Processing
        lstm_out, _ = self.lstm1(features)  # (batch_size, seq_len, 256)
        lstm_out, (h_n, c_n) = self.lstm2(lstm_out)  # (batch_size, seq_len, 128)
        
        # Letzten Hidden State verwenden (bidirectional: forward + backward)
        final_features = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, 128) - beide Richtungen
        
        # Klassifikation
        output = self.classifier(final_features)
        return output
class TemporalBiLSTMModel(nn.Module):
    def __init__(self, num_classes, sequence_length):
        super(TemporalBiLSTMModel, self).__init__()
        self.sequence_length = sequence_length

        # Vortrainiertes CNN (ResNet18 oder MobileNetV2)
        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.cnn_base = backbone.features  # Entfernt Classifier

        # CNN-Feature-Ausgabe: 1280 Kanäle bei MobileNetV2
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 1280, 1, 1)
            nn.Flatten(),                  # (B, 1280)
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )

        # Bidirektionale LSTM-Schichten
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)

        # Klassifikation
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),  # BiLSTM: 2 * hidden_size
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, 3, H, W)
        B, T, C, H, W = x.shape

        # Merge Batch und Zeit
        x = x.view(B * T, C, H, W)

        # CNN Feature Extraktion pro Frame
        x = self.cnn_base(x)  # → (B*T, 1280, h', w')
        x = self.feature_extractor(x)  # → (B*T, 256)

        # Zurück zu Sequenzstruktur
        x = x.view(B, T, -1)  # (B, T, 256)

        # BiLSTM
        lstm_out, (h_n, _) = self.lstm(x)  # h_n: (num_layers * 2, B, hidden)

        # Nutze letzte Hidden States beider Richtungen
        h_forward = h_n[-2, :, :]  # (B, 128)
        h_backward = h_n[-1, :, :]  # (B, 128)
        h_combined = torch.cat((h_forward, h_backward), dim=1)  # (B, 256)

        # Klassifikation
        out = self.classifier(h_combined)
        return out

    def unfreeze_base_model(self):
        """CNN fine-tuning aktivieren"""
        for param in self.cnn_base.parameters():
            param.requires_grad = True

class TemporalTransferModel(nn.Module):
    """
    NEU: Temporal CNN mit Transfer Learning Base (MobileNetV2)
    """
    def __init__(self, num_classes, sequence_length):
        super(TemporalTransferModel, self).__init__()
        self.sequence_length = sequence_length
        
        # Vortrainierte MobileNetV2 Basis
        mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')  # Korrigiert: weights statt pretrained
        # Entferne den Klassifikator
        self.base_features = mobilenet.features
        
        # Friere die Basis-Features zunächst ein
        for param in self.base_features.parameters():
            param.requires_grad = False
        
        # Feature-Extraktor für jedes Frame
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 256),  # MobileNetV2 hat 1280 Output-Features
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        # Temporal Processing mit bidirektionalem LSTM
        self.lstm1 = nn.LSTM(256, 128, batch_first=True, bidirectional=True, dropout=0.3)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True, dropout=0.3)
        
        # Klassifikations-Schichten
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),  # 128 für bidirectional
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, 3, H, W)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape für CNN: (batch_size * sequence_length, 3, H, W)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        
        # MobileNetV2 Feature Extraktion
        features = self.base_features(x)  # (batch_size * seq_len, 1280, H', W')
        features = self.feature_extractor(features)  # (batch_size * seq_len, 256)
        
        # Zurück zu Sequenz-Format: (batch_size, sequence_length, 256)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM Processing
        lstm_out, _ = self.lstm1(features)  # (batch_size, seq_len, 256)
        lstm_out, (h_n, c_n) = self.lstm2(lstm_out)  # (batch_size, seq_len, 128)
        
        # Letzten Hidden State verwenden (bidirectional)
        final_features = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, 128)
        
        # Klassifikation
        output = self.classifier(final_features)
        return output
    
    def unfreeze_base_model(self):
        """Entsperre die Basis-Features für Fine-Tuning"""
        for param in self.base_features.parameters():
            param.requires_grad = True


# Beispiel für Verwendung:
def create_data_transforms():
    """Erstelle Daten-Transformationen"""
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet Normalisierung
    ])
    return transform


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device=None, save_name="temporal_model.pth"):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    best_model_path = os.path.abspath(os.path.join("models", "best_model_finetuned.pth"))

    os.makedirs("models", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for sequences, _, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # ===== Validation =====
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sequences, _, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total

        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # ===== Modell mit bester Val-Acc speichern =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f" Neues bestes Modell gespeichert ({val_acc:.2f}%) unter:\n{best_model_path}")

    print(" Training abgeschlossen.")

    # ===== Modell nach dem letzten Epoch zusätzlich speichern =====
    final_model_path = os.path.abspath(os.path.join("models", save_name))
    torch.save(model.state_dict(), final_model_path)
    print(f" Letztes Modell gespeichert unter:\n{final_model_path}")

if __name__ == "__main__":
    transform = create_data_transforms()

    train_dataset = VideoSequenceDataset(
        train_dir='dataset/train',
        img_size=(224, 224),
        sequence_length=10,
        transform=transform
    )

    val_dataset = VideoSequenceDataset(
        train_dir='dataset/val',
        img_size=(224, 224),
        sequence_length=10,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, drop_last=True)

    num_classes = len(train_dataset.unique_labels)

    # Modell auswählen
    model = TemporalBiLSTMModel(num_classes=num_classes, sequence_length=10)
    model.unfreeze_base_model()

    # Training
    print("Starte Training...")
    train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001, save_name="temporal_bilstm_model.pth")
