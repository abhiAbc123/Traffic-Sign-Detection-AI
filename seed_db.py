import os
import cv2
from ultralytics import YOLO
import database

def seed():
    database.init_db()
    model_path = os.path.join('imageRecognition', 'yolov8m.pt')
    input_dir = os.path.join('imageRecognition', 'input_images')

    if not os.path.exists(model_path) or not os.path.exists(input_dir):
        print("Missing model or input directory.")
        return

    model = YOLO(model_path)
    images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    junctions = database.get_junctions()

    print(f"Seeding database with detections from {len(images)} sample images...")
    for idx, img_name in enumerate(images):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model(img, conf=0.25, verbose=False)
        detections = []
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                detections.append({
                    "class_id": cls_id,
                    "class_name": model.names.get(cls_id, str(cls_id)),
                    "confidence": conf
                })

        # Distribute across available junctions
        j_id = junctions[idx % len(junctions)]['id']
        timing_rec = database.record_detection_and_timing(
            img_name,
            detections,
            inference_ms=150.0,
            junction_id=j_id
        )
        print(f"[{idx+1}/{len(images)}] {img_name} -> Junction {j_id}: {len(detections)} obj | Green: {timing_rec['green_duration_sec']}s | {timing_rec['congestion_level']}")

    print("Seeding completed successfully!")

if __name__ == '__main__':
    seed()

