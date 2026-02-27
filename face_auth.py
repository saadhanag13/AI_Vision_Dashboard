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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = DeepFace.represent(
                img_path=frame,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=False
            )

            for face in results:

                embedding = face["embedding"]
                facial_area = face["facial_area"]

                similarity = cosine_similarity(
                    embedding,
                    known_embedding
                )

                percentage = round(similarity * 100, 2)

                if similarity >= THRESHOLD:
                    label = f"Authorized - {percentage}%"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown - {percentage}%"
                    color = (0, 0, 255)

                x = facial_area["x"]
                y = facial_area["y"]
                w = facial_area["w"]
                h = facial_area["h"]

                cv2.rectangle(frame, (x, y),
                              (x+w, y+h), color, 2)

                cv2.putText(frame, label,
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

        except Exception as e:
            print("Error:", e)

        cv2.imshow("Face Authentication", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()