import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd

LABELS = ["fokussiert", "abgelenkt", "handy", "nicht_anwesend"]
model = load_model("models/focus_model.h5")

log = []
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (150, 150)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    label = LABELS[np.argmax(pred)]

    cv2.putText(frame, f"Status: {label}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Productivity Tracker - Dummy Test", frame)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.append([timestamp, label])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Log speichern
df = pd.DataFrame(log, columns=["timestamp", "label"])
df.to_csv("logs/test_log.csv", index=False)
