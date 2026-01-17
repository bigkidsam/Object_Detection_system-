import streamlit as st
import cv2
import time
from ultralytics import YOLO
import numpy as np

# ---------- CONFIG ----------
DEVELOPER_NAME = "Swayam Singh (Developer)"
STUDENT_ID = "CA/DE1/3358"
INTERNSHIP = "Artificial Intelligence Internship"
SPONSOR = "Code Alpha"

st.set_page_config(page_title="YOLO Object Detection & Tracking", layout="wide")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------- SIMPLE TRACKER (UNCHANGED) ----------
class SimpleTracker:
    def __init__(self, max_lost=10, iou_threshold=0.3):
        self.next_track_id = 0
        self.tracks = {}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def iou(self, A, B):
        xA, yA = max(A[0], B[0]), max(A[1], B[1])
        xB, yB = min(A[2], B[2]), min(A[3], B[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (A[2] - A[0]) * (A[3] - A[1])
        areaB = (B[2] - B[0]) * (B[3] - B[1])
        return inter / (areaA + areaB - inter)

    def update(self, detections):
        updated = {}
        used = set()

        for tid, t in self.tracks.items():
            best_iou, best_idx = 0, -1
            for i, d in enumerate(detections):
                if i in used:
                    continue
                iou = self.iou(t["bbox"], d["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= self.iou_threshold:
                updated[tid] = detections[best_idx]
                used.add(best_idx)
            else:
                t["missed"] += 1
                if t["missed"] <= 10:
                    updated[tid] = t

        for i, d in enumerate(detections):
            if i not in used:
                updated[self.next_track_id] = d
                self.next_track_id += 1

        self.tracks = updated
        return [{"id": k, **v} for k, v in self.tracks.items()]

tracker = SimpleTracker()

# ---------- STREAMLIT UI ----------
st.title("📸 YOLOv8 Object Detection & Tracking")
st.markdown(f"""
**Developer:** {DEVELOPER_NAME}  
**Student ID:** {STUDENT_ID}  
**Internship:** {INTERNSHIP}  
**Sponsor:** {SPONSOR}
""")

start = st.button("▶ Start Camera")
stop = st.button("⏹ Stop Camera")

frame_placeholder = st.empty()
fps_placeholder = st.empty()

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False

# ---------- CAMERA LOOP ----------
cap = cv2.VideoCapture(0)
prev_time = time.time()

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.error("Webcam not accessible")
        break

    results = model(frame, conf=0.4, verbose=False)
    detections = []

    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            label = model.names[int(b.cls[0])]
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "missed": 0
            })

    tracked = tracker.update(detections)

    for obj in tracked:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {obj['id']} {obj['label']}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    fps_placeholder.markdown(f"### FPS: `{fps:.2f}`")
    frame_placeholder.image(frame, channels="BGR")

cap.release()
