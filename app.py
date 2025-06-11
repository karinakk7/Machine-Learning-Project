from flask import Flask, render_template, Response, jsonify, request
from scripts.live_feedback import ProductivityMonitor  
import cv2
import threading
import io
from contextlib import redirect_stdout

app = Flask(__name__)
monitor = ProductivityMonitor()
camera = cv2.VideoCapture(0)

# Globale Variablen für Steuerung
is_running = False
is_paused = False
monitor_thread = None
camera_active = False

def generate_frames():
    global is_running, is_paused, camera_active
    while is_running and camera_active:
        if not is_paused:
            success, frame = camera.read()
            if not success:
                break
            else:
                activity, confidence, predictions = monitor.get_stable_prediction(frame)
                monitor.log_activity(activity, confidence)
                monitor.give_feedback(activity, confidence)
                frame = monitor.draw_ui(frame, activity, confidence, predictions)

                # Kodieren für HTML-Stream
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Pausiert - zeige leeren Frame oder Pausenbild
            import numpy as np
            pause_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(pause_frame, "PAUSIERT", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            ret, buffer = cv2.imencode('.jpg', pause_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_monitoring():
    global is_running, is_paused, camera_active
    if not is_running:
        is_running = True
        is_paused = False
        camera_active = True
        return jsonify({'status': 'started', 'message': 'Monitoring gestartet!'})
    return jsonify({'status': 'already_running', 'message': 'Monitoring läuft bereits!'})

@app.route('/pause', methods=['POST'])
def pause_monitoring():
    global is_paused
    is_paused = not is_paused
    status = 'paused' if is_paused else 'resumed'
    message = 'Monitoring pausiert!' if is_paused else 'Monitoring fortgesetzt!'
    return jsonify({'status': status, 'message': message})

@app.route('/stop', methods=['POST'])
def stop_monitoring():
    global is_running, is_paused, camera_active
    is_running = False
    is_paused = False
    camera_active = False
    return jsonify({'status': 'stopped', 'message': 'Monitoring gestoppt!'})

@app.route('/report')
def report():
    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            monitor.generate_session_report()
        report_text = buffer.getvalue()
        return jsonify({'report': report_text, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/feedback')
def get_feedback():
    try:
        # Hole die letzten Feedback-Nachrichten
        feedback_messages = getattr(monitor, 'recent_feedback', [])
        return jsonify({'feedback': feedback_messages, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/reset', methods=['POST'])
def reset():
    try:
        monitor.reset_statistics()
        return jsonify({'message': 'Statistiken zurückgesetzt!', 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/status')
def get_status():
    global is_running, is_paused
    return jsonify({
        'is_running': is_running,
        'is_paused': is_paused,
        'camera_active': camera_active
    })

if __name__ == '__main__':
    app.run(debug=True)