import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

DEVELOPER_NAME = "Swayam Singh (Developer)"
STUDENT_ID = "CA/DE1/3358"
INTERNSHIP = "Artificial Intelligence Internship"
SPONSOR = "Code Alpha"

WINDOW_NAME = "Task 4 - Object Detection & Tracking"


class SimpleTracker:
    def __init__(self, max_lost=10, iou_threshold=0.3):
        self.next_track_id = 0
        self.tracks = {}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea)    

    def update(self, detections):
        updated_tracks = {}
        used_det_idx = set()

        for track_id, track in self.tracks.items():
            best_iou = 0.0
            best_det_idx = -1

            for i, det in enumerate(detections):
                if i in used_det_idx:
                    continue
                iou_val = self.iou(track['bbox'], det['bbox'])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det_idx = i

            if best_iou >= self.iou_threshold and best_det_idx != -1:
                updated_tracks[track_id] = {
                    'bbox': detections[best_det_idx]['bbox'],
                    'label': detections[best_det_idx]['label'],
                    'missed': 0
                }
                used_det_idx.add(best_det_idx)
            else:
                track['missed'] += 1
                if track['missed'] <= self.max_lost:
                    updated_tracks[track_id] = track

        for i, det in enumerate(detections):
            if i not in used_det_idx:
                updated_tracks[self.next_track_id] = {
                    'bbox': det['bbox'],
                    'label': det['label'],
                    'missed': 0
                }
                self.next_track_id += 1

        self.tracks = updated_tracks

        return [
            {'id': tid, 'bbox': t['bbox'], 'label': t['label']}
            for tid, t in self.tracks.items()
        ]


def load_yolo_model():
    model = YOLO("yolov8n.pt")  # lightweight & fast
    return model


def detect_objects(model, frame):
    results = model(frame, conf=0.4, iou=0.45, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            label = model.names[cls]
            score = float(box.conf[0])

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'label': label,
                'score': score
            })

    return detections


def draw_tracks(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        label = obj['label']
        track_id = obj['id']

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}: {label}",
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def draw_overlay_info(frame, fps):
    info = [
        f"Developer: {DEVELOPER_NAME}",
        f"ID: {STUDENT_ID} | {INTERNSHIP}",
        f"Sponsor: {SPONSOR}",
        f"FPS: {fps:.1f}"
    ]

    y = 20
    for line in info:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        y += 18


def main():
    print("===============================================")
    print(" TASK 4: Object Detection & Tracking ")
    print("===============================================")
    print(f"Developer : {DEVELOPER_NAME}")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Internship: {INTERNSHIP}")
    print(f"Sponsor   : {SPONSOR}")
    print("Press 'q' to quit.")
    print("===============================================")

    model = load_yolo_model()
    tracker = SimpleTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(model, frame)
        tracked_objects = tracker.update(detections)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-8)
        prev_time = curr_time

        draw_tracks(frame, tracked_objects)
        draw_overlay_info(frame, fps)
        
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
