import threading
import time
import cv2
import base64
import json
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# --- Shared State mit Modellintegration ---
class SessionState:
    def __init__(self,
                 model_path="models/best_model_finetuned.keras",
                 class_indices_path="models/class_indices.json"):
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Fehler Modell laden: {e}")
            self.model = None
        try:
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            self.index_to_class = {v: k for k, v in class_indices.items()}
        except Exception:
            self.index_to_class = {0: "abgelenkt", 1: "fokussiert",
                                   2: "handy", 3: "nicht_anwesend"}
        self.classes = list(self.index_to_class.values())
        self.img_size = (224,224)
        self.confidence_threshold = 0.7

        self.prediction_buffer = deque(maxlen=10)
        self.capture = None
        self.thread = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.start_time = None

        self.activity_log = []
        self.distracted_count = 0
        self.focus_to_distract_count = 0
        self.last_activity = None

    def preprocess(self, frame):
        img = cv2.resize(frame, self.img_size)
        img = img_to_array(img) / 255.0
        return np.expand_dims(img,0)

    def get_stable_prediction(self, frame):
        if not self.model:
            return None, 0, None
        preds = self.model.predict(self.preprocess(frame), verbose=0)[0]
        idx = int(np.argmax(preds))
        cls = self.index_to_class.get(idx, 'unbekannt')
        conf = float(preds[idx])
        self.prediction_buffer.append((cls, conf))
        if len(self.prediction_buffer) >= 3:
            recent = [p[0] for p in list(self.prediction_buffer)[-3:]]
            cls = max(set(recent), key=recent.count)
        return cls, conf, preds

    def log_activity(self, activity, confidence):
        ts = datetime.now()
        self.activity_log.append({'timestamp': ts,
                                  'activity': activity,
                                  'confidence': confidence})

        # Wechsel von "fokussiert" zu "abgelenkt"/"handy"
        if self.last_activity == 'fokussiert' and activity in ['abgelenkt', 'handy']:
            self.focus_to_distract_count += 1

        if activity in ['abgelenkt', 'handy'] and confidence > self.confidence_threshold:
            self.distracted_count += 1

        self.last_activity = activity

    def draw_ui(self, frame, activity, confidence):
        h, w = frame.shape[:2]
        colors = {
            'fokussiert':      (0,200,0),
            'abgelenkt':       (255,165,0),
            'handy':           (255,0,0),
            'nicht_anwesend':  (128,128,128)
        }
        col = colors.get(activity, (255,255,255))
        cv2.rectangle(frame, (0,0), (w-1,h-1), col, 4)
        cv2.putText(frame, f"{activity.upper()} ({confidence:.2f})", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        elapsed = int((datetime.now() - self.start_time).total_seconds())
        cv2.putText(frame, f"Zeit: {elapsed//60:02d}:{elapsed%60:02d}",
                    (10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255,255,255), 2)
        return frame

    def start(self):
        if self.running:
            return
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Kamera konnte nicht geöffnet werden.")
            return
        self.running = True
        self.start_time = datetime.now()
        self.activity_log.clear()
        self.distracted_count = 0
        self.focus_to_distract_count = 0
        self.last_activity = None
        self.prediction_buffer.clear()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.capture:
            self.capture.release()

    def _update(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue
            cls, conf, _ = self.get_stable_prediction(frame)
            if cls:
                self.log_activity(cls, conf)
            frame = self.draw_ui(frame, cls or '---', conf or 0)
            with self.lock:
                self.frame = frame.copy()
            time.sleep(0.03)

    def get_frame_encoded(self):
        with self.lock:
            if self.frame is None:
                return None
            _, buf = cv2.imencode('.jpg', self.frame)
            return "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

    def generate_report(self):
        total = (datetime.now() - self.start_time).total_seconds()
        durations = defaultdict(float)
        for i in range(len(self.activity_log)-1):
            dt = (self.activity_log[i+1]['timestamp'] -
                  self.activity_log[i]['timestamp']).total_seconds()
            durations[self.activity_log[i]['activity']] += dt
        if self.activity_log:
            durations[self.activity_log[-1]['activity']] += (
                datetime.now() - self.activity_log[-1]['timestamp']
            ).total_seconds()
        data = [{'Activity': act, 'Duration': sec/60}
                for act, sec in durations.items()]
        focused = durations.get('fokussiert', 0)
        score = (focused / total * 100) if total > 0 else 0
        return data, total/60, self.distracted_count, score

state = SessionState()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([

    html.Script('''
        function notifyUser(message) {
            if (!("Notification" in window)) return;
            if (Notification.permission === "granted") {
                new Notification(message);
            }
        }
        document.addEventListener('DOMContentLoaded', () => {
            if ("Notification" in window && Notification.permission === "default") {
                Notification.requestPermission();
            }
        });
    '''),

    dcc.Store(id='stop-trigger', data=0),
    dcc.Store(id='focus-to-distract', data=0),  # Ablenkungswechsel-Zähler
    dcc.Store(id='last-notified', data=0),      # Für Notification bei 5er-Schritten
    html.Div(id='js-trigger', style={'display':'none'}),

    dbc.Row([
        dbc.Col(html.H2("Produktivitäts-Monitor"), width=8),
        dbc.Col(html.Div([
            dbc.Button("Start", id="start-btn", color="success", n_clicks=0, className="me-2"),
            dbc.Button("Stop",  id="stop-btn",  color="danger",  n_clicks=0)
        ]), width=4, style={'textAlign':'right'})
    ], align="center", className="my-3"),

    dbc.Row([
        dbc.Col(dbc.Card(
            dbc.CardBody(html.Img(id="video-stream", style={"width":"100%"})),
            className="shadow"
        ), width=8),
        dbc.Col(dcc.Graph(id='activity-graph'), width=4)
    ]),

    dcc.Interval(id="interval", interval=1000//30, n_intervals=0, disabled=True),
    dcc.Interval(id="notify-interval", interval=1000, n_intervals=0, disabled=True),

    dbc.Row(dbc.Col(html.Div(id="report-container"), className="mt-4"))

], fluid=True)

# --- Callbacks ---

@app.callback(
    Output('interval', 'disabled'),
    Output('notify-interval', 'disabled'),
    Input('start-btn','n_clicks'),
    Input('stop-btn', 'n_clicks'),
    State('interval','disabled'),
    State('notify-interval','disabled')
)
def control(start_clicks, stop_clicks, disabled, notify_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    btn = ctx.triggered[0]['prop_id'].split('.')[0]
    if btn == 'start-btn':
        state.start()
        return False, False  # beide Intervalle aktivieren
    else:
        state.stop()
        return True, True    # beide Intervalle deaktivieren

@app.callback(
    Output('video-stream','src'),
    Input('interval','n_intervals')
)
def update_frame(n):
    src = state.get_frame_encoded()
    return src or dash.no_update

@app.callback(
    Output('activity-graph','figure'),
    Input('interval','n_intervals')
)
def update_graph(n):
    data, _, _, _ = state.generate_report()
    fig = go.Figure([go.Bar(
        x=[d['Activity'] for d in data],
        y=[d['Duration'] for d in data]
    )])
    fig.update_layout(title='Aktivitätsverteilung (Min)', yaxis_title='Minuten')
    return fig

@app.callback(
    Output('stop-trigger','data'),
    Input('stop-btn','n_clicks'),
    State('stop-trigger','data')
)
def bump_stop(n, current):
    if n == 0:
        raise dash.exceptions.PreventUpdate
    return current + 1

# Fokus-to-distract count regelmäßig aktualisieren
@app.callback(
    Output('focus-to-distract', 'data'),
    Input('interval', 'n_intervals')
)
def update_focus_to_distract_store(n):
    return state.focus_to_distract_count

# Clientside Notification bei jedem 5. Ablenkungswechsel
app.clientside_callback(
    """
    function(n_intervals, focusToDistract, lastNotified) {
        if (Notification.permission !== 'granted') return lastNotified;
        if (focusToDistract >= 5 && focusToDistract % 5 === 0 && focusToDistract > lastNotified) {
            new Notification('Ablenkungswechsel: ' + focusToDistract);
            return focusToDistract;
        }
        return lastNotified;
    }
    """,
    Output('last-notified', 'data'),
    Input('notify-interval', 'n_intervals'),
    State('focus-to-distract', 'data'),
    State('last-notified', 'data')
)

@app.callback(
    Output('report-container','children'),
    Input('stop-btn','n_clicks')
)
def show_report(n):
    if n == 0:
        raise dash.exceptions.PreventUpdate
    data, duration, distracts, score = state.generate_report()
    return dbc.Card(dbc.CardBody([
        html.H4("Session Report", className="card-title"),
        html.P(f"Dauer: {duration:.1f} Min"),
        html.P(f"Ablenkungen: {distracts}"),
        html.P(f"Ablenkungs-Wechsel (fokussiert → abgelenkt/handy): {state.focus_to_distract_count}"),
        html.P(f"Produktivitäts-Score: {score:.1f}%")
    ]), className="shadow")

if __name__ == '__main__':
    app.run(debug=True)
