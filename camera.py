# activate venv: source venv/bin/activate
import cv2
from matplotlib import pyplot as plt

# cap = cv2.VideoCapture(0)

# ret, frame = cap.read()

# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.show()
# cap.release()


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
