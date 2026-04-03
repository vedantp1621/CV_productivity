import time
import queue as q

from ultralytics import YOLO

from background.shared import (
    yolo_queue, monitoring_active,
    detection_state, detection_lock,
    session_stats, session_lock,
    phone_boxes, boxes_lock,
)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def phone_loop():
    model = _get_model()
    last_t         = None
    phone_detected = False   # previous frame's result, used for time accounting

    print("Phone: thread started.")

    while monitoring_active.is_set():
        try:
            frame = yolo_queue.get(timeout=1)
        except q.Empty:
            last_t = None
            continue

        now = time.monotonic()
        if last_t is not None:
            dt = now - last_t
            if phone_detected:
                with session_lock:
                    session_stats["phone_seconds"] += dt
        last_t = now

        results      = model(frame, conf=0.4, classes=[67], verbose=False)
        raw_boxes    = results[0].boxes
        phone_detected = len(raw_boxes) > 0

        # Save bounding boxes for frame annotation
        if phone_detected:
            bxs = raw_boxes.xyxy.cpu().numpy().astype(int).tolist()
            cfs = raw_boxes.conf.cpu().numpy().tolist()
        else:
            bxs, cfs = [], []

        with boxes_lock:
            phone_boxes["boxes"] = bxs
            phone_boxes["confs"] = cfs

        with detection_lock:
            detection_state["phone"] = phone_detected

    with boxes_lock:
        phone_boxes["boxes"] = []
        phone_boxes["confs"] = []
    with detection_lock:
        detection_state["phone"] = False
    print("Phone: thread stopped.")
