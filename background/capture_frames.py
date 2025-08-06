import cv2
import os
import time
from datetime import datetime


def capture_images_every_second(save_folder="database"):
    
    os.makedirs(save_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.png"
            filepath = os.path.join(save_folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filepath}")

            time.sleep(1)

    finally:
        cap.release()
        cv2.destroyAllWindows()
