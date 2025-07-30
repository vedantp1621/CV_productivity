from flask import Flask
import os
import shutil
import config

def create_app(config_class="config.Config"):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Clear the captures folder at startup
    captures_folder = app.config["UPLOAD_FOLDER"]
    if os.path.exists(captures_folder):
        shutil.rmtree(captures_folder)
    os.makedirs(captures_folder)

    from .routes import main
    app.register_blueprint(main)

    return app
