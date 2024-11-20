from flask import Flask, Response, render_template
import cv2
import os

app = Flask(__name__)

# Path to overlay image
OVERLAY_IMAGE = "khalil.png"

@app.route("/")
def index():
    # Main page
    return render_template("ar_overlay.html")

@app.route("/video_feed")
def video_feed():
    def generate():
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        overlay = cv2.imread(OVERLAY_IMAGE, -1)  # Load the overlay image
        cap = cv2.VideoCapture(0)  # Access the default webcam

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Apply overlay to each detected face
            for (x, y, w, h) in faces:
                resized_overlay = cv2.resize(overlay, (w, h))  # Resize overlay to match face size
                frame = overlay_on_face(frame, resized_overlay, x, y)

            # Encode frame as JPEG for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to apply overlay
def overlay_on_face(frame, overlay, x, y):
    h, w, _ = overlay.shape

    # Add alpha channel if missing
    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):  # RGB Channels
        frame[y:y+h, x:x+w, c] = (alpha_s * overlay[:, :, c] + alpha_l * frame[y:y+h, x:x+w, c])
    return frame

if __name__ == "__main__":
    app.run(debug=True)
