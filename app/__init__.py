from flask import Flask
import os
import config

def create_app(config_class="config.Config"):
    app = Flask(__name__)
    app.config.from_object(config_class)

    from .routes import main
    app.register_blueprint(main)

    return app
