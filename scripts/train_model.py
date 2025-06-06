import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# Konfiguration
train_dir = 'dataset/train'
val_dir = 'dataset/val'
model_save_path = 'models/focus_model_transfer.keras'

img_size = (224, 224)
batch_size = 16
initial_epochs = 20
fine_tune_epochs = 10

# Erstelle models Ordner falls er nicht existiert
os.makedirs('models', exist_ok=True)

print("=== Produktivitäts-Monitor Training ===")
print(f"Training Dir: {train_dir}")
print(f"Validation Dir: {val_dir}")

# Erweiterte Datenvorbereitung mit mehr Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1],
    shear_range=0.1,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Daten-Generatoren
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir, 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode='categorical',
    shuffle=False
)

# Klasseninformationen anzeigen
print(f"\nAnzahl Trainingsbilder: {train_generator.samples}")
print(f"Anzahl Validierungsbilder: {val_generator.samples}")
print(f"Gefundene Klassen: {list(train_generator.class_indices.keys())}")
print(f"Anzahl Klassen: {train_generator.num_classes}")

# Klassenindizes speichern für spätere Verwendung
class_indices = train_generator.class_indices
with open('models/class_indices.json', 'w') as f:
    json.dump(class_indices, f, indent=2)
print("Klassenindizes gespeichert in models/class_indices.json")

# Basis-Modell laden
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(*img_size, 3)
)

# Basis-Modell einfrieren
base_model.trainable = False

# Modell aufbauen mit mehr Layern für bessere Performance
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(train_generator.num_classes, activation='softmax')
])

# Modell kompilieren
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nModell-Architektur:")
model.summary()

# Callbacks für besseres Training
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'models/best_model_initial.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\n=== Phase 1: Initial Training ===")
# Erstes Training
history_initial = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n=== Phase 2: Fine-Tuning ===")
# Fine-Tuning: Basis-Modell teilweise entfrosten
base_model.trainable = True

# Nur die letzten Layer des Basis-Modells trainieren
fine_tune_at = len(base_model.layers) - 20

# Alle Layer bis fine_tune_at einfrieren
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Mit sehr kleiner Lernrate neu kompilieren
model.compile(
    optimizer=Adam(learning_rate=0.0001/10),  # 10x kleinere Lernrate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-Tuning Callbacks
fine_tune_callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'models/best_model_finetuned.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Fine-Tuning Training
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=len(history_initial.history['loss']),
    validation_data=val_generator,
    callbacks=fine_tune_callbacks,
    verbose=1
)

# Trainingsverlauf kombinieren
def combine_histories(hist1, hist2):
    combined = {}
    for key in hist1.history.keys():
        combined[key] = hist1.history[key] + hist2.history[key]
    return combined

combined_history = combine_histories(history_initial, history_fine)

# Finales Modell speichern
model.save(model_save_path)
print(f"\nFinales Modell gespeichert: {model_save_path}")

# Trainingsverlauf visualisieren
def plot_training_history(history, title="Training History"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy Plot
    ax1.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss Plot
    ax2.plot(history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Markiere den Übergang zwischen Initial Training und Fine-Tuning
    ax1.axvline(x=initial_epochs-1, color='red', linestyle='--', alpha=0.7, label='Fine-Tuning Start')
    ax2.axvline(x=initial_epochs-1, color='red', linestyle='--', alpha=0.7, label='Fine-Tuning Start')
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(combined_history)

# Modell evaluieren
print("\n=== Modell Evaluation ===")

# Vorhersagen auf Validation Set
val_generator.reset()  # Generator zurücksetzen
predictions = model.predict(val_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# True Labels extrahieren
true_classes = val_generator.classes[:len(predicted_classes)]

# Klassennamen für Report
class_names = list(train_generator.class_indices.keys())

# Classification Report
print("\nClassification Report:")
print(classification_report(
    true_classes, 
    predicted_classes, 
    target_names=class_names,
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Detaillierte Accuracy pro Klasse
print("\nAccuracy pro Klasse:")
for i, class_name in enumerate(class_names):
    class_mask = true_classes == i
    if np.sum(class_mask) > 0:
        class_accuracy = np.sum(predicted_classes[class_mask] == i) / np.sum(class_mask)
        print(f"{class_name}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")

# Finale Statistiken
final_val_accuracy = history_fine.history['val_accuracy'][-1]
final_val_loss = history_fine.history['val_loss'][-1]

print(f"\n=== Training Abgeschlossen ===")
print(f"Finale Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Finale Validation Loss: {final_val_loss:.4f}")

print(f"\nGespeicherte Dateien:")
print(f"- {model_save_path} (Finales Modell)")
print(f"- models/class_indices.json (Klassenindizes)")
print(f"- models/training_history.png (Trainingsverlauf)")
print(f"- models/confusion_matrix.png (Confusion Matrix)")
print(f"- models/best_model_initial.h5 (Bestes Modell nach Initial Training)")
print(f"- models/best_model_finetuned.h5 (Bestes Modell nach Fine-Tuning)")

# Funktion zum Testen einzelner Bilder
def predict_single_image(model_path, image_path, class_indices_path='models/class_indices.json'):
    """
    Funktion zum Testen des Modells mit einem einzelnen Bild
    """
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import json
    
    # Modell und Klassenindizes laden
    model = load_model(model_path)
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Index zu Klassennamen Mapping
    index_to_class = {v: k for k, v in class_indices.items()}
    
    # Bild laden und preprocessieren
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Vorhersage
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = index_to_class[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    
    print(f"Vorhersage: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Alle Wahrscheinlichkeiten:")
    for i, prob in enumerate(prediction[0]):
        print(f"  {index_to_class[i]}: {prob:.4f}")
    
    return predicted_class, confidence
