# Machine-Learning-Project

## Projektteam: Efficio

Alexander Rohr (1533656), Luis Litters(4512765), Tim Stelzner(3482360) \& Karina Krebs(8369760)

## Projektbeschreibung

Ziel dieses Projekts ist die Entwicklung eines Systems zur automatisierten Erkennung des Aufmerksamkeitszustands einer Person während der Bildschirmarbeit. Mithilfe von Videoaufnahmen und modernen Deep-Learning-Architekturen werden drei Zustände klassifiziert:

- **Fokussiertes Arbeiten**
- **Ablenkung durch Smartphone**
- **Allgemeine Ablenkung (z. B. Gespräch mit anderen Personen)**

Diese Klassifikation erfolgt durch die Analyse von Videodaten, die über eine Arbeitsplatzkamera aufgezeichnet werden. Die Anwendung verfolgt das Ziel, neue Formen der adaptiven Arbeitsplatzgestaltung und des Self-Monitorings zu ermöglichen.

## Verwendete Technologien

- **Programmiersprache**: Python 3.x
- **Frameworks**:
  - TensorFlow / Keras
  - OpenCV (Videoverarbeitung)
  - NumPy, Pandas (Datenhandling)
  - Matplotlib / Seaborn (Visualisierung)
- **Modellarchitekturen**:
  - Reise-CNN (eigene Architektur zur Merkmalsextraktion)
  - Transfer Learning CNNs (z. B. MobileNetV2, EfficientNet)
  - Bidirectional LSTM (für Sequenzklassifikation)

