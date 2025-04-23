import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from playsound import playsound
import threading
import os
from datetime import datetime
import tkinter as tk
from PIL import Image, ImageTk

# Load model and setup
model = load_model('mask_detector_modelv5.h5')
save_dir = 'unmasked_faces'
os.makedirs(save_dir, exist_ok=True)

# Mediapipe setup
mp_face_detection = mp.solutions.face_detection

# Alert sound thread
def play_alert():
    playsound('assets_alarm.mp3')

# GUI App
class MaskDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Mask Detection")
        self.root.geometry("800x600")

        # Video Frame
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack()

        # Log Box
        self.log_box = tk.Text(self.root, height=10, width=100)
        self.log_box.pack(pady=10)

        # Start video
        self.cap = cv2.VideoCapture(0)
        self.face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self.update_video()

    def log(self, message):
        self.log_box.insert(tk.END, f"{message}\n")
        self.log_box.see(tk.END)

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)
                x, y = max(0, x), max(0, y)

                face_img = frame[y:y+bh, x:x+bw]
                if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                    continue

                face_resized = cv2.resize(face_img, (100, 100))
                face_norm = face_resized / 255.0
                prediction = model.predict(np.expand_dims(face_norm, axis=0))[0][0]

                label = "No Mask" if prediction < 0.5 else "Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                if label == "No Mask":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    cv2.imwrite(f"{save_dir}/unmasked_{timestamp}.jpg", face_img)
                    self.log(f"[ALERT] No Mask detected at {timestamp}")

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Convert to image and show in GUI
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def __del__(self):
        self.cap.release()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MaskDetectionApp(root)
    root.mainloop()
