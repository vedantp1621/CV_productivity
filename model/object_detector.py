from ultralytics import YOLO

def yolo_detection():
    model = YOLO('yolov8n.pt')

    results = model(source=0, show=True, stream=True, conf=0.4)
    for r in results:
        pass  # or analyze r.boxes, r.names, etc.
