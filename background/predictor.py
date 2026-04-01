import queue as q

from background.shared import yolo_queue, monitoring_active, detection_state, detection_lock
from model.predictor import detect_phone


def yolo_loop():
    print("YOLO: thread started.")
    while monitoring_active.is_set():
        try:
            frame = yolo_queue.get(timeout=1)
        except q.Empty:
            continue

        phone_detected = detect_phone(frame)
        with detection_lock:
            detection_state["phone"] = phone_detected

    # Clear state when stopped
    with detection_lock:
        detection_state["phone"] = False
    print("YOLO: thread stopped.")
