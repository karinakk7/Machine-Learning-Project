from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

train_dir = 'dataset/train'
val_dir = 'dataset/val'

img_size = (224, 224)  # MobileNetV2 erwartet 224x224
batch_size = 16

# Datenvorbereitung
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   rotation_range=20,
                                   zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Basis-Modell laden, ohne den oberen Klassifizierer (include_top=False)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))

# Basis-Modell einfrieren (kein Training der vortrainierten Gewichte)
base_model.trainable = False

# Neues Modell bauen
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(4, activation='softmax')  # 4 Klassen: fokussiert, abgelenkt, handy, nicht_anwesend
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Optional: Basis-Modell entfrosten und feintunen
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5),  # kleiner Lernrate fürs Feintuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

model.save('models/focus_model_transfer.h5')
