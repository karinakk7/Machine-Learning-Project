import cv2
import numpy as np
import json
import time
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from collections import defaultdict, deque

class ProductivityMonitor:
    def __init__(self, model_path="models/best_model_finetuned.keras", 
                 class_indices_path="models/class_indices.json"):
        """
        Initialisiert den Produktivitäts-Monitor
        
        Args:
            model_path: Pfad zum trainierten Modell
            class_indices_path: Pfad zu den Klassenindizes
        """
        print("Lade Produktivitäts-Monitor...")
        
        # Initialize basic attributes first (before potential early return)
        self.img_size = (224, 224)
        self.check_interval = 10  # Sekunden zwischen Checks
        self.distraction_threshold = 3  # Nach 3 Ablenkungen → Pause vorschlagen
        self.confidence_threshold = 0.7  # Mindest-Confidence für Feedback
        self.cap = None
        self.model = None
        
        # Initialize statistics
        self.reset_statistics()
        
        # Modell laden
        try:
            self.model = load_model(model_path)
            print(f"✅ Modell geladen: {model_path}")
        except Exception as e:
            print(f" Fehler beim Laden des Modells: {e}")
            print("  Monitor wird ohne Modell fortgesetzt (nur für Tests)")
            # Don't return - continue initialization
        
        # Klassenindizes laden
        try:
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            # Index zu Klassenname Mapping erstellen
            self.index_to_class = {v: k for k, v in class_indices.items()}
            self.classes = list(class_indices.keys())
            print(f" Klassen geladen: {self.classes}")
        except Exception as e:
            print(f" Fehler beim Laden der Klassenindizes: {e}")
            # Fallback auf Standard-Klassen
            self.classes = ["abgelenkt", "fokussiert", "handy", "nicht_anwesend"]
            self.index_to_class = {i: name for i, name in enumerate(self.classes)}
            print(f"📝 Verwende Standard-Klassen: {self.classes}")
        
    def reset_statistics(self):
        """Setzt alle Statistiken zurück"""
        self.start_time = time.time()
        self.last_check_time = 0
        self.distracted_count = 0
        self.consecutive_distractions = 0
        
        # Detaillierte Statistiken
        self.session_stats = {
            'fokussiert': [],
            'abgelenkt': [],
            'handy': [],
            'nicht_anwesend': []
        }
        
        # Zeitbasierte Logs
        self.activity_log = []
        
        # Ringpuffer für letzte 10 Vorhersagen (für Stabilität)
        self.prediction_buffer = deque(maxlen=10)
        
    def preprocess_frame(self, frame):
        """Preprocessiert ein Frame für das Modell"""
        img = cv2.resize(frame, self.img_size)
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
        
    def get_stable_prediction(self, frame):
        """
        Macht eine Vorhersage und stabilisiert sie durch Pufferung
        """
        # Fallback wenn kein Modell geladen ist
        if self.model is None:
            # Simuliere eine zufällige Vorhersage für Tests
            import random
            predicted_class = random.choice(self.classes)
            confidence = random.uniform(0.5, 0.95)
            all_predictions = np.random.rand(len(self.classes))
            all_predictions = all_predictions / all_predictions.sum()  # Normalisieren
            
            self.prediction_buffer.append((predicted_class, confidence))
            return predicted_class, confidence, all_predictions
        
        img = self.preprocess_frame(frame)
        prediction = self.model.predict(img, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        predicted_class = self.index_to_class[predicted_class_idx]
        
        # Zur Pufferung hinzufügen
        self.prediction_buffer.append((predicted_class, confidence))
        
        # Stabilisierte Vorhersage durch Mehrheitsentscheidung
        if len(self.prediction_buffer) >= 3:
            recent_predictions = [pred[0] for pred in list(self.prediction_buffer)[-3:]]
            # Mehrheitsentscheidung
            from collections import Counter
            most_common = Counter(recent_predictions).most_common(1)[0][0]
            return most_common, confidence, prediction[0]
        
        return predicted_class, confidence, prediction[0]
        
    def log_activity(self, activity, confidence):
        """Loggt Aktivität mit Zeitstempel"""
        timestamp = datetime.now()
        self.activity_log.append({
            'timestamp': timestamp,
            'activity': activity,
            'confidence': confidence
        })
        
        # Statistiken aktualisieren
        if activity in self.session_stats:
            self.session_stats[activity].append(timestamp)
        
    def give_feedback(self, current_activity, confidence):
        """Gibt kontextuelles Feedback basierend auf der aktuellen Aktivität"""
        current_time = time.time()
        
        # Nur alle check_interval Sekunden Feedback geben
        if current_time - self.last_check_time < self.check_interval:
            return
            
        self.last_check_time = current_time
        
        # Nur bei ausreichender Confidence Feedback geben
        if confidence < self.confidence_threshold:
            return
            
        # Feedback basierend auf Aktivität
        if current_activity == "fokussiert":
            self.consecutive_distractions = 0
            print("✅ Sehr gut! Du arbeitest konzentriert weiter.")
            
        elif current_activity == "abgelenkt":
            self.consecutive_distractions += 1
            self.distracted_count += 1
            
            if self.consecutive_distractions == 1:
                print("⚠️  Du scheinst abgelenkt zu sein - versuche dich wieder zu fokussieren!")
            elif self.consecutive_distractions == 2:
                print("🎯 Fokus! Lass dich nicht weiter ablenken.")
            else:
                print("🔄 Du bist schon länger abgelenkt. Brauchst du eine kurze Pause?")
                
        elif current_activity == "handy":
            self.consecutive_distractions += 1
            self.distracted_count += 1
            
            print("📱 Handy weggelegt! Konzentriere dich auf deine Arbeit.")
            if self.consecutive_distractions >= 2:
                print("📵 Vielleicht das Handy stumm schalten oder weglegen?")
                
        elif current_activity == "nicht_anwesend":
            print("👤 Du scheinst nicht am Arbeitsplatz zu sein.")
            
        # Pause vorschlagen nach mehreren Ablenkungen
        if self.distracted_count > 0 and self.distracted_count % self.distraction_threshold == 0:
            print("⏸️  Du warst öfter abgelenkt. Wie wäre es mit einer 5-Minuten-Pause?")
            print("🧘 Kurz entspannen kann die Konzentration verbessern!")
            
    def draw_ui(self, frame, current_activity, confidence, all_predictions):
        """Zeichnet die Benutzeroberfläche auf das Frame"""
        height, width = frame.shape[:2]
        
        # Farbe basierend auf Aktivität
        colors = {
            'fokussiert': (0, 255, 0),      # Grün  
            'abgelenkt': (0, 165, 255),     # Orange
            'handy': (0, 0, 255),           # Rot
            'nicht_anwesend': (128, 128, 128) # Grau
        }
        
        color = colors.get(current_activity, (255, 255, 255))
        
        # Model status anzeigen
        model_status = "✅ Modell aktiv" if self.model is not None else "⚠️ Demo-Modus"
        cv2.putText(frame, model_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Hauptstatus anzeigen
        cv2.putText(frame, f"Status: {current_activity}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Confidence anzeigen
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Alle Vorhersagewahrscheinlichkeiten anzeigen
        y_offset = 140
        for i, class_name in enumerate(self.classes):
            if i < len(all_predictions):
                prob = all_predictions[i]
                text = f"{class_name}: {prob:.2f}"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25
            
        # Sessionstatistiken
        session_time = time.time() - self.start_time
        minutes = int(session_time // 60)
        seconds = int(session_time % 60)
        
        cv2.putText(frame, f"Session: {minutes:02d}:{seconds:02d}", 
                   (width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Ablenkungen: {self.distracted_count}", 
                   (width - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Rahmen um das Bild basierend auf Status
        cv2.rectangle(frame, (0, 0), (width-1, height-1), color, 5)
        
        return frame
        
    def generate_session_report(self):
        """Generiert einen Bericht über die Session"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*50)
        print("📊 PRODUKTIVITÄTSBERICHT")
        print("="*50)
        
        print(f"🕐 Gesamte Session-Zeit: {total_time/60:.1f} Minuten")
        print(f"🎯 Gesamte Ablenkungen: {self.distracted_count}")
        
        # Zeitverteilung berechnen
        if self.activity_log:
            activity_durations = defaultdict(float)
            
            for i in range(len(self.activity_log) - 1):
                current = self.activity_log[i]
                next_activity = self.activity_log[i + 1]
                duration = (next_activity['timestamp'] - current['timestamp']).total_seconds()
                activity_durations[current['activity']] += duration
            
            # Letzte Aktivität bis jetzt
            if self.activity_log:
                last_activity = self.activity_log[-1]
                duration = (datetime.now() - last_activity['timestamp']).total_seconds()
                activity_durations[last_activity['activity']] += duration
            
            print("\n📈 Zeitverteilung:")
            for activity, duration in activity_durations.items():
                percentage = (duration / total_time) * 100
                print(f"  {activity}: {duration/60:.1f} min ({percentage:.1f}%)")
            
            # Produktivitätsscore
            focused_time = activity_durations.get('fokussiert', 0)
            productivity_score = (focused_time / total_time) * 100
            print(f"\n🏆 Produktivitäts-Score: {productivity_score:.1f}%")
            
            if productivity_score >= 80:
                print("🌟 Ausgezeichnet! Du warst sehr produktiv!")
            elif productivity_score >= 60:
                print("👍 Gut gemacht! Aber da geht noch mehr.")
            elif productivity_score >= 40:
                print("⚠️  Okay, aber versuche weniger Ablenkungen zu haben.")
            else:
                print("🔄 Da ist noch Luft nach oben. Versuche dich besser zu fokussieren.")
        else:
            print("📝 Keine Aktivitätsdaten verfügbar.")
        
        print("="*50)
        
    def run(self):
        """Hauptschleife des Monitors"""
        print("\n🎯 Produktivitäts-Monitor gestartet!")
        if self.model is None:
            print("⚠️  Läuft im Demo-Modus (kein Modell geladen)")
        print("Drücke 'q' zum Beenden, 'r' zum Zurücksetzen der Statistiken")
        print("Drücke 's' für einen Zwischenbericht")
        print("-" * 50)
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Kamera konnte nicht geöffnet werden!")
            return
            
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Frame konnte nicht gelesen werden.")
                    break
                
                # Vorhersage machen
                current_activity, confidence, all_predictions = self.get_stable_prediction(frame)
                
                # Aktivität loggen
                self.log_activity(current_activity, confidence)
                
                # Feedback geben
                self.give_feedback(current_activity, confidence)
                
                # UI zeichnen
                frame = self.draw_ui(frame, current_activity, confidence, all_predictions)
                
                # Frame anzeigen
                cv2.imshow("Produktivitäts-Monitor", frame)
                
                # Tastatureingaben
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("\n🔄 Statistiken zurückgesetzt!")
                    self.reset_statistics()
                elif key == ord('s'):
                    self.generate_session_report()
                    
        except KeyboardInterrupt:
            print("\n⏹️  Monitor durch Benutzer gestoppt.")
        except Exception as e:
            print(f"\n❌ Fehler aufgetreten: {e}")
            
        finally:
            # Abschlussbericht
            self.generate_session_report()
            
            # Aufräumen
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("\n👋 Auf Wiedersehen!")

# Hauptprogramm
if __name__ == "__main__":
    # Monitor erstellen und starten
    monitor = ProductivityMonitor(
        model_path="models/best_model_finetuned.keras",
        class_indices_path="models/class_indices.json"
    )
    
    # Monitor starten
    monitor.run()