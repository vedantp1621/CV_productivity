# activate venv:   source venv/bin/activate
 
import cv2
from matplotlib import pyplot as plt

def start_camera():
    print("hello  this is running")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
