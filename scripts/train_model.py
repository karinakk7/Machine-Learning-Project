import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torchvision.models as models
import seaborn as sns
from scipy import signal
from scipy.fft import fft
import pickle


class EnhancedFocusTrainer:
    def __init__(self, train_dir='dataset/train', val_dir='dataset/val', 
                 model_save_dir='models/', img_size=(224, 224)):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.model_save_dir = model_save_dir
        self.img_size = img_size
        self.batch_size = 32
        
        # Modell-Typen
        self.model_types = {
            'transfer_learning': self.build_transfer_learning_model,
            'frequency_analysis': self.build_frequency_model,
            'temporal_cnn': self.build_improved_temporal_cnn,  # Verbesserte Version
            'temporal_transfer': self.build_temporal_transfer_model,  # NEU: Transfer Learning + Temporal
            'hybrid_model': self.build_hybrid_model,
            'lstm_features': self.build_lstm_feature_model
        }
        
        os.makedirs(model_save_dir, exist_ok=True)
    
    def extract_frequency_features(self, image_sequence):
        """
        Extrahiert Frequenz-Features aus einer Bildsequenz
        
        Args:
            image_sequence: Liste von Bildern (Frames aus Video)
        """
        features = []
        
        for img in image_sequence:
            # Zu Grayscale konvertieren
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # 1. DCT (Discrete Cosine Transform) Features
            dct_features = cv2.dct(np.float32(gray))
            dct_energy = np.sum(dct_features**2)
            
            # 2. FFT Features - Frequenzspektrum
            fft_img = fft(gray.flatten())
            fft_magnitude = np.abs(fft_img)
            
            # Frequenzbänder analysieren
            low_freq = np.sum(fft_magnitude[:len(fft_magnitude)//4])
            mid_freq = np.sum(fft_magnitude[len(fft_magnitude)//4:len(fft_magnitude)//2])
            high_freq = np.sum(fft_magnitude[len(fft_magnitude)//2:3*len(fft_magnitude)//4])
            
            # 3. Kantendetektion (Edge density als Fokus-Indikator)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # 4. Textur-Features (LBP - Local Binary Patterns)
            lbp_features = self.calculate_lbp_features(gray)
            
            # 5. Optischer Fluss (wenn mehrere Frames verfügbar)
            optical_flow_features = self.calculate_optical_flow_features(image_sequence)
            
            frame_features = [
                dct_energy,
                low_freq, mid_freq, high_freq,
                edge_density,
                *lbp_features,
                *optical_flow_features
            ]
            
            features.append(frame_features)
        
        return np.array(features)
    
    def calculate_lbp_features(self, gray_img):
        """Local Binary Pattern Features"""
        # Vereinfachte LBP Implementation
        lbp = np.zeros_like(gray_img)
        
        for i in range(1, gray_img.shape[0]-1):
            for j in range(1, gray_img.shape[1]-1):
                center = gray_img[i, j]
                binary = 0
                
                # 8 Nachbarn prüfen
                neighbors = [
                    gray_img[i-1, j-1], gray_img[i-1, j], gray_img[i-1, j+1],
                    gray_img[i, j+1], gray_img[i+1, j+1], gray_img[i+1, j],
                    gray_img[i+1, j-1], gray_img[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        binary += 2**k
                
                lbp[i, j] = binary
        
        # Histogramm als Features
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return hist / np.sum(hist)  # Normalisiert
    
    def calculate_optical_flow_features(self, image_sequence):
        """Optischer Fluss zwischen aufeinanderfolgenden Frames"""
        if len(image_sequence) < 2:
            return [0, 0, 0, 0]  # Default values
        
        # Letzten zwei Frames nehmen
        prev_frame = cv2.cvtColor(image_sequence[-2], cv2.COLOR_BGR2GRAY) if len(image_sequence[-2].shape) == 3 else image_sequence[-2]
        curr_frame = cv2.cvtColor(image_sequence[-1], cv2.COLOR_BGR2GRAY) if len(image_sequence[-1].shape) == 3 else image_sequence[-1]
        
        # Optischer Fluss berechnen
        flow = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, 
                                       np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2),
                                       None)[0]
        
        if flow is not None and len(flow) > 0:
            flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            return [
                np.mean(flow_magnitude),
                np.std(flow_magnitude),
                np.max(flow_magnitude),
                np.sum(flow_magnitude > 1.0)  # Anzahl signifikanter Bewegungen
            ]
        
        return [0, 0, 0, 0]
    
    def load_video_sequences(self, sequence_length=10):
        """
        Lädt Bildsequenzen aus den Trainingsordnern
        
        Args:
            sequence_length: Anzahl aufeinanderfolgender Frames pro Sequenz
        """
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
            
            # Weniger überlappende Sequenzen für bessere Diversität
            step_size = max(1, sequence_length // 4)
            # Sequenzen erstellen
            for i in range(0, len(image_files) - sequence_length * 3 + 1, step_size):
                sequence = []
                
                # Sampling über 3x so viele Frames
                sampling_indices = np.linspace(i, i + sequence_length * 3 - 1, sequence_length, dtype=int)

                for idx in sampling_indices:
                    if idx < len(image_files):
                        img_path = os.path.join(class_path, image_files[idx])
                        img = cv2.imread(img_path)
                        if img is not None:
             
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
                        except:
                            # Fallback: Dummy-Features
                            X_frequency.append(np.zeros(260))  # Standard Feature-Größe
                        
                        y_labels.append(class_name)
        
        print(f"Geladene Sequenzen: {len(X_sequences)}")
        return np.array(X_sequences), np.array(X_frequency), np.array(y_labels)
    
def build_transfer_learning_model(self, num_classes):
    base_model = models.mobilenet_v2(pretrained=True)
    for param in base_model.features.parameters():
        param.requires_grad = False

    model = nn.Sequential(
        base_model.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.BatchNorm1d(1280),
        nn.Dropout(0.4),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    return model


def build_improved_temporal_cnn(num_classes, sequence_length, img_size=(224, 224)):
    H, W = img_size

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.sequence_length = sequence_length

            # TimeDistributed Conv-Blocks (als 2D, über Zeitachse geschleift)
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
            )
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
            )
            self.conv_block3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
            )

            # GlobalAveragePooling + Dense pro Frame
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.frame_dense = nn.Sequential(
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5)
            )

            # LSTM
            self.lstm1 = nn.LSTM(
                input_size=256,
                hidden_size=128,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )
            self.lstm2 = nn.LSTM(
                input_size=256,
                hidden_size=64,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )

            # Klassifikations-Teil
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            B, T, C, H, W = x.size()
            frame_features = []

            for t in range(T):
                xt = x[:, t]                          # (B, C, H, W)
                out = self.conv_block1(xt)
                out = self.conv_block2(out)
                out = self.conv_block3(out)
                out = self.global_pool(out)           # (B, 128, 1, 1)
                out = out.view(B, -1)                 # (B, 128)
                out = self.frame_dense(out)           # (B, 256)
                frame_features.append(out)

            x_seq = torch.stack(frame_features, dim=1)  # (B, T, 256)

            x_seq, _ = self.lstm1(x_seq)  # (B, T, 256)
            x_seq, _ = self.lstm2(x_seq)  # (B, T, 128)
            x_last = x_seq[:, -1, :]      # (B, 128)

            return self.classifier(x_last)

    return Model()


def build_temporal_transfer_model(num_classes, sequence_length, img_size=(224, 224)):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.sequence_length = sequence_length

            # MobileNetV2 als Feature-Extraktor (pro Frame)
            mobilenet = models.mobilenet_v2(pretrained=True).features
            for param in mobilenet.parameters():
                param.requires_grad = False
            self.feature_extractor = nn.Sequential(
                mobilenet,
                nn.AdaptiveAvgPool2d(1),  # GlobalAveragePooling2D
            )
            self.feature_dim = 1280  # MobileNetV2-Ausgabe vor Kopf

            self.frame_dense = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3)
            )

            # LSTM (bidirectional)
            self.lstm1 = nn.LSTM(
                input_size=256,
                hidden_size=128,
                batch_first=True,
                dropout=0.3,
                bidirectional=True
            )
            self.lstm2 = nn.LSTM(
                input_size=256,
                hidden_size=64,
                batch_first=True,
                dropout=0.3,
                bidirectional=True
            )

            # Klassifikation
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            B, T, C, H, W = x.size()
            features = []

            for t in range(T):
                xt = x[:, t]                            # (B, C, H, W)
                out = self.feature_extractor(xt)        # (B, 1280, 1, 1)
                out = out.view(B, -1)                   # (B, 1280)
                out = self.frame_dense(out)             # (B, 256)
                features.append(out)

            x_seq = torch.stack(features, dim=1)        # (B, T, 256)
            x_seq, _ = self.lstm1(x_seq)                # (B, T, 256)
            x_seq, _ = self.lstm2(x_seq)                # (B, T, 128)
            x_last = x_seq[:, -1, :]                    # letztes Zeitschritt

            return self.classifier(x_last)

    return Model()

    
def build_frequency_model(num_classes, frequency_feature_dim):
    model = nn.Sequential(
        nn.Linear(frequency_feature_dim, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),

        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.4),

        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),

        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(64, num_classes)
    )
    return model

    
def build_temporal_cnn(num_classes, sequence_length, img_size=(224, 224)):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(16)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(32)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),  # GlobalAveragePooling2D
                nn.Dropout(0.3)
            )

            self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, dropout=0.3, bidirectional=False)
            self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, dropout=0.3, bidirectional=False)

            self.classifier = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            B, T, C, H, W = x.shape
            features = []

            for t in range(T):
                xt = x[:, t]               # (B, C, H, W)
                xt = self.conv1(xt)
                xt = self.conv2(xt)
                xt = self.conv3(xt)
                xt = xt.view(B, -1)        # (B, 64)
                features.append(xt)

            x_seq = torch.stack(features, dim=1)  # (B, T, 64)
            x_seq, _ = self.lstm1(x_seq)          # (B, T, 64)
            x_seq, _ = self.lstm2(x_seq)          # (B, T, 32)
            x_last = x_seq[:, -1, :]              # (B, 32)

            return self.classifier(x_last)

    return Model()
    
from torchvision.models import mobilenet_v2

def build_hybrid_model(num_classes, frequency_feature_dim, img_size=(224, 224)):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            mobilenet = mobilenet_v2(pretrained=True).features
            for p in mobilenet.parameters():
                p.requires_grad = False

            self.image_branch = nn.Sequential(
                mobilenet,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(1280, 256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

            self.freq_branch = nn.Sequential(
                nn.Linear(frequency_feature_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU()
            )

            self.classifier = nn.Sequential(
                nn.Linear(256 + 64, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes)
            )

        def forward(self, img, freq):
            x1 = self.image_branch(img)
            x2 = self.freq_branch(freq)
            x = torch.cat([x1, x2], dim=1)
            return self.classifier(x)

    return Model()
import torch
import torch.nn as nn

def build_lstm_feature_model(num_classes, sequence_length, feature_dim):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.lstm1 = nn.LSTM(
                input_size=feature_dim,
                hidden_size=128,
                batch_first=True,
                bidirectional=False
            )
            self.dropout1 = nn.Dropout(0.3)

            self.lstm2 = nn.LSTM(
                input_size=128,
                hidden_size=64,
                batch_first=True,
                bidirectional=False
            )
            self.dropout2 = nn.Dropout(0.3)

            self.lstm3 = nn.LSTM(
                input_size=64,
                hidden_size=32,
                batch_first=True,
                bidirectional=False
            )

            self.classifier = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):  # x: (B, T, F)
            x, _ = self.lstm1(x)     # (B, T, 128)
            x = self.dropout1(x)

            x, _ = self.lstm2(x)     # (B, T, 64)
            x = self.dropout2(x)

            x, _ = self.lstm3(x)     # (B, T, 32)
            x = x[:, -1, :]          # letzter Zeitschritt → (B, 32)

            return self.classifier(x)

    return Model()

    
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import json

def train_model(self, model_type='temporal_cnn', epochs=30):
    print(f"\n=== Training {model_type} Model ===")
    
    if model_type in ['temporal_cnn', 'temporal_transfer', 'lstm_features', 'hybrid_model', 'frequency_analysis']:
        X_sequences, X_frequency, y_labels = self.load_video_sequences()
        if len(X_sequences) == 0:
            print("FEHLER: Keine Daten geladen!")
            return None, None

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_labels)
        num_classes = len(label_encoder.classes_)
        with open(f'{self.model_save_dir}class_indices_{model_type}.json', 'w') as f:
            json.dump({i: c for i, c in enumerate(label_encoder.classes_)}, f, indent=2)

        if model_type == 'frequency_analysis':
            X_train, X_val, y_train, y_val = train_test_split(X_frequency, y_encoded, stratify=y_encoded, test_size=0.2)
            model = build_frequency_model(num_classes, X_frequency.shape[1])

        elif model_type == 'temporal_cnn':
            X_train, X_val, y_train, y_val = train_test_split(X_sequences, y_encoded, stratify=y_encoded, test_size=0.2)
            model = build_improved_temporal_cnn(num_classes, X_sequences.shape[1], self.img_size)

        elif model_type == 'temporal_transfer':
            X_train, X_val, y_train, y_val = train_test_split(X_sequences, y_encoded, stratify=y_encoded, test_size=0.2)
            model = build_temporal_transfer_model(num_classes, X_sequences.shape[1], self.img_size)

        elif model_type == 'lstm_features':
            X_freq_seq = []
            for seq in X_sequences:
                seq_uint8 = [(frame * 255).astype(np.uint8) for frame in seq]
                seq_feat = self.extract_frequency_features(seq_uint8)
                X_freq_seq.append(seq_feat)
            X_freq_seq = np.array(X_freq_seq)
            X_train, X_val, y_train, y_val = train_test_split(X_freq_seq, y_encoded, stratify=y_encoded, test_size=0.2)
            model = build_lstm_feature_model(num_classes, X_freq_seq.shape[1], X_freq_seq.shape[2])

        elif model_type == 'hybrid_model':
            X_images = X_sequences[:, 0]
            X_train_seq, X_val_seq, X_train_freq, X_val_freq, y_train, y_val = train_test_split(
                X_images, X_frequency, y_encoded, stratify=y_encoded, test_size=0.2)
            model = build_hybrid_model(num_classes, X_sequences.shape[1], X_frequency.shape[1], self.img_size)

        # Training vorbereiten
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005 if model_type in ['temporal_cnn', 'temporal_transfer'] else 0.001)
        batch_size = self.batch_size

        # Hybrid separat behandelt
        if model_type == 'hybrid_model':
            train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                          torch.tensor(X_train_freq, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long))
            val_dataset = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32),
                                        torch.tensor(X_val_freq, dtype=torch.float32),
                                        torch.tensor(y_val, dtype=torch.long))
        else:
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_acc = 0
        history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

        for epoch in range(epochs):
            model.train()
            total, correct, running_loss = 0, 0, 0.0
            for batch in train_loader:
                if model_type == 'hybrid_model':
                    X1, X2, y = [b.to(device) for b in batch]
                    outputs = model(X1, X2)
                else:
                    X, y = [b.to(device) for b in batch]
                    outputs = model(X)

                loss = loss_fn(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * y.size(0)
                total += y.size(0)
                correct += (outputs.argmax(1) == y).sum().item()

            train_acc = correct / total
            train_loss = running_loss / total

            # Validierung
            model.eval()
            total, correct, val_loss_total = 0, 0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if model_type == 'hybrid_model':
                        X1, X2, y = [b.to(device) for b in batch]
                        outputs = model(X1, X2)
                    else:
                        X, y = [b.to(device) for b in batch]
                        outputs = model(X)
                    loss = loss_fn(outputs, y)
                    val_loss_total += loss.item() * y.size(0)
                    total += y.size(0)
                    correct += (outputs.argmax(1) == y).sum().item()

            val_acc = correct / total
            val_loss = val_loss_total / total

            print(f"[Epoch {epoch+1}] Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")

            history["accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)
            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"{self.model_save_dir}best_{model_type}_model.pt")

        torch.save(model.state_dict(), f"{self.model_save_dir}final_{model_type}_model.pt")
        return model, history

    else:
        print("Standardmodell nicht unterstützt (nur PyTorch-Modelle).")
        return None, None

if __name__ == "__main__":
    trainer = EnhancedFocusTrainer(
        train_dir='dataset/train',
        val_dir='dataset/val',
        model_save_dir='models/'
    )

    model_types = [
        #'temporal_cnn',         
         'temporal_transfer',
        # 'frequency_analysis',
        # 'lstm_features',
        # 'hybrid_model'
    ]

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*60}")
        
        try:
            model, history = trainer.train_model(model_type=model_type, epochs=30)
            print(f" {model_type} erfolgreich trainiert!")
        except Exception as e:
            print(f" Fehler beim Training von {model_type}: {e}")

