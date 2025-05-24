# activate venv: source venv/bin/activate

# sources:
## https://medium.com/@amit25173/opencv-eye-tracking-aeb4f1b46aa3
 
import cv2
from matplotlib import pyplot as plt

# Simple code to start camera
# def start_camera():
   
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
       
#         ret, frame = cap.read()
#         cv2.imshow('Webcam', frame) 

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# Eye tracking software (using Haar cascade classifier for face detection)
def eye_tracking():
    
    # Pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Pre-trained eye detection model
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Connect to proper camera port
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw shapes around faces and eyes
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_color = roi_color[ey:ey+eh, ex:ex+ew]

                # Thresholding to isolate the pupil
                _, thresh = cv2.threshold(eye_gray, 70, 255, cv2.THRESH_BINARY_INV)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                
                if contours:
                    (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
                    cx, cy, radius = int(cx), int(cy), int(radius)
                    cv2.circle(eye_color, (cx, cy), radius, (255, 0, 0), 2)

                    # Only draw lines if cx, cy are defined correctly
                    cv2.line(frame, (cx, cy), (cx + 50, cy), (0, 255, 0), 2)  # Horizontal line
                    cv2.line(frame, (cx, cy), (cx, cy + 50), (0, 255, 0), 2)  # Vertical line

        cv2.imshow('Eye Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    eye_tracking()
