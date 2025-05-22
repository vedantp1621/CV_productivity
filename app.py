from flask import Flask
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/cam")
def camera():
    subprocess.Popen(["python3", "camera.py"])  # runs independently
    print("process shouldve ran")
    return "Camera launched"

if __name__ == "__main__":
    app.run(debug=True)
