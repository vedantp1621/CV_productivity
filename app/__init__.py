from flask import Flask
import os
import config

def create_app():
    app = Flask(__name__)

    # Load config if needed
    app.config.from_pyfile("../config.py", silent=True)

    # Create training_data directory
    os.makedirs(config.TRAINING_DATA_DIR, exist_ok=True)

    # Register routes
    from .routes import init_routes

    init_routes(app)

    return app
