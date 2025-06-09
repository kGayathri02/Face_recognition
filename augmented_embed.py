import os
import cv2
import pickle
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from pathlib import Path

# ---------- Settings ----------
AADHAAR_IMG_PATH = "gayu_a_ch.jpeg"  # Updated to use the correct image file
AUG_DIR = "augmented_faces"
EMBEDDING_FILE = "aadhaar_embeddings.pkl"
SIMULATED_YAW_ANGLES = [-45, -30, -15, 0, 15, 30, 45]

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def augment_yaw_sim(image, angles):
    """Simulate face rotation by flipping and brightness change"""
    augmented = []
    for yaw in angles:
        if yaw == 0:
            aug_img = image.copy()
            augmented.append((f"yaw_0", aug_img))
        elif yaw < 0:
            aug_img = cv2.convertScaleAbs(cv2.flip(image, 1), alpha=1.0 + abs(yaw)/100)
            augmented.append((f"yaw_{yaw}", aug_img))
        else:
            aug_img = cv2.convertScaleAbs(image, alpha=1.0 - yaw/100)
            augmented.append((f"yaw_{yaw}", aug_img))
    return augmented

def generate_augmented_images(img_path, output_dir):
    # Check if image exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ Image file not found: {img_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Failed to load image: {img_path}. Check if it's a valid image file.")
    
    print(f"✅ Successfully loaded image: {img_path}")
    print(f"📐 Image dimensions: {img.shape}")
    
    # Generate augmented images
    print("🔄 Generating augmented images...")
    augmented_images = augment_yaw_sim(img, SIMULATED_YAW_ANGLES)
    
    # Save augmented images
    saved_paths = []
    for name, aug_img in augmented_images:
        output_path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(output_path, aug_img)
        saved_paths.append(output_path)
        print(f"💾 Saved augmented image: {output_path}")
    
    return saved_paths

def register_embeddings(app, img_paths):
    embeddings = {}
    for path in img_paths:
        img = cv2.imread(path)
        faces = app.get(img)
        if faces:
            embeddings[Path(path).stem] = faces[0].embedding
            print(f"✅ Registered: {Path(path).stem}")
        else:
            print(f"⚠️ No face found in: {path}")
    
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings

if __name__ == "__main__":
    # Initialize face analysis
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    
    print("🔄 Generating yaw-augmented images...")
    paths = generate_augmented_images(AADHAAR_IMG_PATH, AUG_DIR)
    
    print("\n📝 Registering face embeddings...")
    embeddings = register_embeddings(app, paths)
    
    print(f"\n✅ Successfully registered {len(embeddings)} face variations.")
