import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

# Modell laden
model = load_model("models/focus_model_transfer.h5")
classes = ["abgelenkt", "fokussiert", "handy", "nicht_anwesend"]

# Kamera starten
cap = cv2.VideoCapture(0)

# Zähler für Ablenkungen
distracted_count = 0
last_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera konnte nicht gelesen werden.")
            break

        # Bild vorbereiten: resize auf 150x150 (statt 224x224)
        img = cv2.resize(frame, (224, 224))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Vorhersage machen
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        label = classes[predicted_class]

        current_time = time.time()

        # Alle 1 Sekunde prüfen
        if current_time - last_time > 10:
            last_time = current_time

            if label != "fokussiert":
                distracted_count += 1
                if label == "abgelenkt":
                    print("❗ Du bist abgelenkt – fokussiere dich wieder.")
                elif label == "handy":
                    print("📱 Leg dein Handy weg!")
                elif label == "nicht_anwesend":
                    print("👤 Du scheinst nicht anwesend zu sein.")

                # Alle 3 Ablenkungen Erinnerung an Pause
                if distracted_count % 10 == 0:
                    print("⏸️ Brauchst du vielleicht eine 5-Minuten-Pause?")
            else:
                distracted_count = 0
                print("✅ Du arbeitest konzentriert.")

        # Status im Videofenster anzeigen (grün = fokussiert, rot = sonst)
        color = (0, 255, 0) if label == "fokussiert" else (0, 0, 255)
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Live Feedback", frame)

        # Mit 'q' beenden
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
