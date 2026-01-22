from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['video']
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, 'output_video.mp4')
        file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        speed_estimate = 0
        alert = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            # Demo speed logic
            num_objects = len(results[0].boxes) if results[0].boxes is not None else 0
            speed_estimate = num_objects * 10  # fake km/h

            if speed_estimate > 60:
                alert = True
                cv2.putText(annotated, "RED ALERT: OVER SPEED!",
                            (40, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

            cv2.putText(annotated, f"Speed: {speed_estimate} km/h",
                        (40, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)

            out.write(annotated)

        cap.release()
        out.release()

        return f"Processing Done. Speed={speed_estimate} km/h | Alert={alert}"

    return '''
    <h2>AI Traffic Speed Detection</h2>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="video" required>
      <br><br>
      <button type="submit">Upload & Analyze</button>
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
