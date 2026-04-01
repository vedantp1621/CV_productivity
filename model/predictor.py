from ultralytics import YOLO

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def detect_phone(frame) -> bool:
    """Return True if a cell phone (class 67) is detected in the frame."""
    model = _load_model()
    results = model(frame, conf=0.4, classes=[67], verbose=False)
    return len(results[0].boxes) > 0
