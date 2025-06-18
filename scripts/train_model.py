import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, 
                                   LSTM, TimeDistributed, Conv1D, MaxPooling1D, Flatten,
                                   Input, Concatenate, Conv2D, MaxPooling2D, GRU, Bidirectional)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import json
import math
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
    
    def build_improved_temporal_cnn(self, num_classes, sequence_length):
        """
        VERBESSERTE Temporal CNN Architektur mit Transfer Learning Features
        """
        model = Sequential([
            Input(shape=(sequence_length, *self.img_size, 3)),
            
            # Erste Conv-Block (mehr Filter, bessere Feature-Extraktion)
            TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Dropout(0.25)),
            
            # Zweite Conv-Block
            TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Dropout(0.25)),
            
            # Dritte Conv-Block
            TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Dropout(0.25)),
            
            # Feature-Extraktion
            TimeDistributed(GlobalAveragePooling2D()),
            TimeDistributed(Dense(256, activation='relu')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(Dropout(0.5)),
            
            # Bidirektionale LSTM für bessere Temporal-Features
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
            Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
            
            # Klassifikations-Schichten
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        return model       
    def build_temporal_transfer_model(self, num_classes, sequence_length):
        """
        NEU: Temporal CNN mit Transfer Learning Base
        """
        # Vortrainierte Basis
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Zunächst einfrieren
        
        # Feature-Extraktor für jedes Frame
        feature_extractor = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3)
        ])
        
        # Temporal Modell
        model = Sequential([
            Input(shape=(sequence_length, *self.img_size, 3)),
            TimeDistributed(feature_extractor),
            
            # Temporal Processing
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)),
            Bidirectional(LSTM(64, dropout=0.3)),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_learning_model(self, num_classes):
        """Dein ursprüngliches Transfer Learning Modell"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                               input_shape=(*self.img_size, 3))
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def build_frequency_model(self, num_classes, frequency_feature_dim):
        """Modell basierend auf Frequenz-Features"""
        model = Sequential([
            Input(shape=(frequency_feature_dim,)),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def build_temporal_cnn(self, num_classes, sequence_length):
        """CNN für Zeitsequenzen - KORRIGIERT"""
        model = Sequential([
            Input(shape=(sequence_length, *self.img_size, 3)),
            
            # Erste Conv-Schicht
            TimeDistributed(Conv2D(16, (3, 3), activation='relu')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(BatchNormalization()),
            
            # Zweite Conv-Schicht
            TimeDistributed(Conv2D(32, (3, 3), activation='relu')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(BatchNormalization()),
            
            # Dritte Conv-Schicht
            TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
            TimeDistributed(GlobalAveragePooling2D()),
            TimeDistributed(Dropout(0.3)),
            
            # LSTM-Schichten
            LSTM(64, return_sequences=True, dropout=0.3),
            LSTM(32, dropout=0.3),
            
            # Dense-Schichten
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def build_lstm_feature_model(self, num_classes, sequence_length, feature_dim):
        """LSTM für Feature-Sequenzen"""
        model = Sequential([
            Input(shape=(sequence_length, feature_dim)),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def build_hybrid_model(self, num_classes, sequence_length, frequency_feature_dim):
        """Hybrid-Modell: Transfer Learning + Frequenz-Features"""
        
        # Transfer Learning Branch
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                               input_shape=(*self.img_size, 3))
        base_model.trainable = False
        
        image_input = Input(shape=(*self.img_size, 3))
        x1 = base_model(image_input)
        x1 = GlobalAveragePooling2D()(x1)
        x1 = Dense(256, activation='relu')(x1)
        x1 = Dropout(0.3)(x1)
        
        # Frequenz Branch
        freq_input = Input(shape=(frequency_feature_dim,))
        x2 = Dense(128, activation='relu')(freq_input)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        x2 = Dense(64, activation='relu')(x2)
        
        # Kombinieren
        combined = Concatenate()([x1, x2])
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.4)(combined)
        output = Dense(num_classes, activation='softmax')(combined)
        
        model = Model(inputs=[image_input, freq_input], outputs=output)
        return model
    
    def train_model(self, model_type='temporal_cnn', epochs=50):
        """
        Trainiert das ausgewählte Modell - VERBESSERT
        """
        print(f"\n=== Training {model_type} Model ===")
        
        if model_type in ['temporal_cnn', 'temporal_transfer', 'lstm_features', 'hybrid_model', 'frequency_analysis']:
            # Video-Sequenzen laden
            X_sequences, X_frequency, y_labels = self.load_video_sequences()
            
            if len(X_sequences) == 0:
                print("FEHLER: Keine Daten geladen!")
                return None, None
            
            # Label Encoding
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_labels)
            y_categorical = to_categorical(y_encoded)
            num_classes = len(label_encoder.classes_)
            
            print(f"Anzahl Klassen: {num_classes}")
            print(f"Klassen: {label_encoder.classes_}")
            print(f"Datenform: {X_sequences.shape}")
            
            # Train/Val Split
            if model_type == 'frequency_analysis':
                X_train, X_val, y_train, y_val = train_test_split(
                    X_frequency, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
                )
                model = self.build_frequency_model(num_classes, X_frequency.shape[1])
                
            elif model_type in ['temporal_cnn', 'temporal_transfer']:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_sequences, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
                )
                if model_type == 'temporal_cnn':
                    model = self.build_improved_temporal_cnn(num_classes, X_sequences.shape[1])
                else:
                    model = self.build_temporal_transfer_model(num_classes, X_sequences.shape[1])
                
            elif model_type == 'lstm_features':
                # Features über Zeit für LSTM vorbereiten
                X_freq_sequences = []
                for seq in X_sequences:
                    seq_uint8 = [(frame * 255).astype(np.uint8) for frame in seq]
                    seq_features = self.extract_frequency_features(seq_uint8)
                    X_freq_sequences.append(seq_features)
                X_freq_sequences = np.array(X_freq_sequences)
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_freq_sequences, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
                )
                model = self.build_lstm_feature_model(num_classes, X_freq_sequences.shape[1], X_freq_sequences.shape[2])
                
            elif model_type == 'hybrid_model':
                # Einzelbilder für Transfer Learning (erstes Bild jeder Sequenz)
                X_images = X_sequences[:, 0]
                
                X_train_seq, X_val_seq, X_train_freq, X_val_freq, y_train, y_val = train_test_split(
                    X_images, X_frequency, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                model = self.build_hybrid_model(num_classes, X_sequences.shape[1], X_frequency.shape[1])
            
            # Klassen-Info speichern
            class_info = {i: class_name for i, class_name in enumerate(label_encoder.classes_)}
            with open(f'{self.model_save_dir}class_indices_{model_type}.json', 'w') as f:
                json.dump(class_info, f, indent=2)
            
        else:
            # Standard Transfer Learning
            return self.train_transfer_learning_model(epochs)
        
        # Modell kompilieren mit optimierter Lernrate
        optimizer = Adam(learning_rate=0.001)
        if model_type in ['temporal_cnn', 'temporal_transfer']:
            optimizer = Adam(learning_rate=0.0005)  # Niedrigere Lernrate für komplexe Modelle
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Modell-Architektur ({model_type}):")
        model.summary()
        
        # Verbesserte Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
            ModelCheckpoint(f'{self.model_save_dir}best_{model_type}_model.keras', 
                          monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        # Training
        try:
            if model_type == 'hybrid_model':
                history = model.fit(
                    [X_train_seq, X_train_freq], y_train,
                    validation_data=([X_val_seq, X_val_freq], y_val),
                    epochs=epochs,
                    batch_size=self.batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=self.batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
        except Exception as e:
            print(f"Training-Fehler: {e}")
            return None, None
        
        # Finales Modell speichern
        model.save(f'{self.model_save_dir}final_{model_type}_model.keras')
        
        # Scaler für Frequenz-Features speichern
        if model_type in ['frequency_analysis', 'hybrid_model']:
            scaler = StandardScaler()
            scaler.fit(X_frequency if model_type == 'frequency_analysis' else X_train_freq)
            with open(f'{self.model_save_dir}frequency_scaler_{model_type}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        
        print(f"\n{model_type} Modell Training abgeschlossen!")
        print(f"Gespeichert: {self.model_save_dir}final_{model_type}_model.keras")
        
        # Training-Verlauf plotten
        self.plot_training_history(history, model_type)
        
        return model, history
    
    def plot_training_history(self, history, model_type):
        """Plottet den Trainingsverlauf"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Accuracy
            ax1.plot(history.history['accuracy'], label='Training Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title(f'{model_type} - Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            
            # Loss
            ax2.plot(history.history['loss'], label='Training Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title(f'{model_type} - Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f'{self.model_save_dir}{model_type}_training_history.png')
            plt.show()
        except Exception as e:
            print(f"Plotting error: {e}")
    
    def train_transfer_learning_model(self, epochs):
        """Transfer Learning Training - Verbessert"""
        
        # Verbesserte Datenaugmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            rotation_range=20,
            zoom_range=0.2,
            width_shift_range=0.15,
            height_shift_range=0.15,
            brightness_range=[0.8, 1.2],
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            self.train_dir, target_size=self.img_size, batch_size=self.batch_size, 
            class_mode='categorical', shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.val_dir, target_size=self.img_size, batch_size=self.batch_size, 
            class_mode='categorical', shuffle=False
        )
        
        model = self.build_transfer_learning_model(train_generator.num_classes)
        model.compile(
            optimizer=Adam(learning_rate=0.001), 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
            ModelCheckpoint(f'{self.model_save_dir}best_transfer_learning_model.keras', 
                          monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        model.save(f'{self.model_save_dir}final_transfer_learning_model.keras')
        
        # Klassen speichern
        with open(f'{self.model_save_dir}class_indices_transfer_learning.json', 'w') as f:
            json.dump(train_generator.class_indices, f, indent=2)
        
        self.plot_training_history(history, 'transfer_learning')
        
        return model, history

# Verwendung
if __name__ == "__main__":
    trainer = EnhancedFocusTrainer(
        train_dir='dataset/train',
        val_dir='dataset/val',
        model_save_dir='models/'
    )
    
    # Verschiedene Modelle trainieren
    model_types = [
        #'temporal_transfer',     # NEU: Transfer Learning + Temporal (empfohlen!)
        'temporal_cnn',         # Verbessertes Temporal CNN
        #'transfer_learning',    # Verbessertes Transfer Learning
        # 'frequency_analysis',   # Nur Frequenz-Features
        # 'lstm_features',       # LSTM auf Feature-Sequenzen
        # 'hybrid_model'         # Kombination (für Experten)
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
    
    print("\n Alle Modelle trainiert und gespeichert!")