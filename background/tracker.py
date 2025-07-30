import cv2
import os
from datetime import datetime


def capture_images_every_second(save_folder="database"):
    """
    Continuously captures one image per second from the webcam and saves it.
    Runs until 'q' is pressed in the OpenCV window.
    Intended to be run in a separate thread.
    """
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

            cv2.imshow("Webcam (Press Q to quit)", frame)

            # Wait 1 second or exit on 'q'
            if cv2.waitKey(1000) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
