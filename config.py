import os

class Config: 
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join("app", "static", "captures")
    MODEL_PATH = os.path.join(BASE_DIR, "model", "saved_model.pkl")
    DISTRACTION_THRESHOLD = 30
    WEBCAM_INDEX = 0
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit
