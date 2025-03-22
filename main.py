import os
import cv2
import base64
import tempfile
from flask import Flask, request, Response, send_from_directory, jsonify
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])
app = Flask(__name__, static_folder='.', static_url_path='')
model = YOLO("best.pt")
globalVideoPath = None
stopStreaming = False

def genFrames():
    global stopStreaming
    if not globalVideoPath:
        return
    cap = cv2.VideoCapture(globalVideoPath)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while True:
        if stopStreaming:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (840, 640))
        results = model(frame)
        annotated = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            break
        frameBytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frameBytes + b'\r\n')
    cap.release()

@app.route('/')
def indexPage():
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])
def uploadFile():
    global globalVideoPath, stopStreaming
    stopStreaming = False
    file = request.files.get('video')
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    file.save(temp.name)
    temp.flush()
    temp.close()
    globalVideoPath = temp.name
    filename = os.path.basename(temp.name)
    return jsonify({"status": "ok", "filename": filename})

@app.route('/start', methods=['POST'])
def startPredictions():
    global globalVideoPath
    if not globalVideoPath:
        return jsonify({"status": "error", "message": "No file set"}), 400
    return jsonify({"status": "ok"})

@app.route('/stop', methods=['POST'])
def stopPredictions():
    global globalVideoPath, stopStreaming
    stopStreaming = True
    if globalVideoPath and os.path.exists(globalVideoPath):
        globalVideoPath = None
    return jsonify({"status": "ok"})

@app.route('/video_feed')
def videoFeed():
    return Response(genFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
