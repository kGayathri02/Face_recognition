import cv2
import pickle
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis

EMBEDDING_FILE = "aadhaar_embeddings.pkl"
THRESHOLD = 0.35  # Lower for more tolerance to yaw

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def recognize_from_camera(app, embeddings):
    cap = cv2.VideoCapture(0)
    print("🟢 Live CCTV started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            emb = face.embedding
            name = "Unknown"
            best_sim = 0

            for aug_name, aug_emb in embeddings.items():
                sim = cosine_similarity(emb, aug_emb)
                if sim > THRESHOLD and sim > best_sim:
                    name = f"MATCH ({aug_name})"
                    best_sim = sim

            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({best_sim:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Aadhaar Multi-Angle Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    with open(EMBEDDING_FILE, "rb") as f:
        embeddings = pickle.load(f)

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    recognize_from_camera(app, embeddings)
