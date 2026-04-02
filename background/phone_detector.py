import queue as q

from ultralytics import YOLO

from background.shared import yolo_queue, monitoring_active, detection_state, detection_lock

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def phone_loop():
    model = _get_model()
    print("Phone: thread started.")

    while monitoring_active.is_set():
        try:
            frame = yolo_queue.get(timeout=1)
        except q.Empty:
            continue

        results = model(frame, conf=0.4, classes=[67], verbose=False)
        phone_detected = len(results[0].boxes) > 0

        with detection_lock:
            detection_state["phone"] = phone_detected

    with detection_lock:
        detection_state["phone"] = False
    print("Phone: thread stopped.")
