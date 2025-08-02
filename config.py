import os

class Config: 
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join("app", "static", "captures")
    FRAME_STORAGE= os.path.join("database")
    MODEL_PATH = os.path.join(BASE_DIR, "model", "saved_model.pkl")
    DISTRACTION_THRESHOLD = 30
    WEBCAM_INDEX = 0
    MAX_CONTENT_LENGTH = 4 * 1024 * 1024 
