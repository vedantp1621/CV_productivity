import cv2
import queue as q

from background.shared import gaze_queue, monitoring_active, detection_state, detection_lock


def gaze_loop():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    print("Gaze: thread started.")
    while monitoring_active.is_set():
        try:
            frame = gaze_queue.get(timeout=1)
        except q.Empty:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            # No face visible — treat as eyes off screen
            with detection_lock:
                detection_state["eyes_off"] = True
            continue

        eyes_found = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                eyes_found = True
                break

        with detection_lock:
            detection_state["eyes_off"] = not eyes_found

    # Clear state when stopped
    with detection_lock:
        detection_state["eyes_off"] = False
    print("Gaze: thread stopped.")
