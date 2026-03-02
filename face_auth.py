# face_auth.py- phone style face auth 

import cv2
import numpy as np
import pickle
import os
from deepface import DeepFace


ENCODING_FILE = "encodings.pkl"
MODEL_NAME = "Facenet"
DETECTOR = "retinaface"
THRESHOLD = 0.75


def normalize(vec):
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)


def cosine_similarity(a, b):
    a = normalize(a)
    b = normalize(b)
    return np.dot(a, b)


# -----------------------------
# REGISTER FACE (from image path)
# -----------------------------
def register_face(image_path):

    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True
    )

    embedding = normalize(result[0]["embedding"])

    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(embedding, f)

    return "Face registered successfully ✅"


# -----------------------------
# VERIFY FACE (Live Camera)
# -----------------------------
def start_face_auth_camera():

    if not os.path.exists(ENCODING_FILE):
        print("No registered face found.")
        return

    with open(ENCODING_FILE, "rb") as f:
        known_embedding = pickle.load(f)

    cap = cv2.VideoCapture(0)

    # FAST detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    frame_count = 0
    last_label = "Detecting..."
    last_color = (255, 255, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # FAST detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        frame_count += 1

        for (x, y, w, h) in faces:

            # Only run DeepFace every 10 frames
            if frame_count % 10 == 0:

                face_crop = frame[y:y+h, x:x+w]

                try:
                    result = DeepFace.represent(
                        img_path=face_crop,
                        model_name=MODEL_NAME,
                        detector_backend="skip",
                        enforce_detection=False
                    )

                    embedding = normalize(result[0]["embedding"])

                    similarity = cosine_similarity(
                        embedding,
                        known_embedding
                    )

                    percentage = round(similarity * 100, 2)

                    if similarity >= THRESHOLD:
                        last_label = f"Authorized - {percentage}%"
                        last_color = (0, 255, 0)
                    else:
                        last_label = f"Unknown - {percentage}%"
                        last_color = (0, 0, 255)

                except Exception:
                    pass

            # Draw always (smooth UI)
            cv2.rectangle(frame, (x, y), (x+w, y+h), last_color, 2)
            cv2.putText(frame, last_label,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        last_color,
                        2)

        cv2.imshow("Face Authentication", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()