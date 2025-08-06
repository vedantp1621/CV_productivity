from ultralytics import YOLO


def yolo_detection():
    model = YOLO("yolov8n.pt")

    for result in model(source=0, stream=True, show=True, conf=0.4, classes=[67]):
        boxes = result.boxes
        if len(boxes) > 0:
            print("Cell phone detected!")
