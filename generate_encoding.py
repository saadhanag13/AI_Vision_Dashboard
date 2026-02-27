# generate_encoding.py

from deepface import DeepFace
import pickle
import numpy as np

def normalize(vec):
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)

img_path = "demo_pic1.jpg"

result = DeepFace.represent(
    img_path=img_path,
    model_name="Facenet",
    detector_backend="retinaface",
    enforce_detection=True
)

embedding = result[0]["embedding"]

# Normalize before saving (VERY IMPORTANT)
embedding = normalize(embedding)

with open("encodings.pkl", "wb") as f:
    pickle.dump(embedding, f)

print("Face registered successfully ✅")