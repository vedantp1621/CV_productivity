import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Folder to store uploaded distracted images
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")

# Path to save the trained model
MODEL_PATH = os.path.join(BASE_DIR, "model", "saved_model.pkl")

# Number of frames to wait before sending distraction alert
DISTRACTION_THRESHOLD = 30

# Webcam device index (0 for default webcam)
WEBCAM_INDEX = 0
