import cv2
from plyer import notification
from model.predictor import is_distracted


def start_tracking():
    cap = cv2.VideoCapture(0)
    distracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if is_distracted(frame):
            distracted_count += 1
        else:
            distracted_count = 0

        if distracted_count > 30:
            notification.notify(
                title="Stay Focused!", message="You seem distracted.", timeout=3
            )
            distracted_count = 0
        print(distracted_count)
