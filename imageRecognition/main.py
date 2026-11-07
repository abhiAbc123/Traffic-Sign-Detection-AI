import argparse
import os
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Traffic Sign and Object Detection")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(base_dir, 'input_images')
    if not os.path.exists(default_input):
        default_input = os.path.join(base_dir, 'DetectionOutput')
    default_output = os.path.join(base_dir, 'DetectionOutput')
    default_model = os.path.join(base_dir, 'yolov8m.pt')

    parser.add_argument('--model', type=str, default=default_model, help='Path to YOLO weights file (.pt)')
    parser.add_argument('--source', type=str, default=default_input, help='Path to input image file or directory')
    parser.add_argument('--output', type=str, default=default_output, help='Directory to save annotated detection output')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    return parser.parse_args()


def process_image(image_path, model, output_path, conf_thresh):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return 0

    results = model(image, conf=conf_thresh, verbose=False)
    detections = 0

    for result in results:
        detections += len(result.boxes)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names.get(cls, str(cls))
            label = f"{class_name} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Label background box for better readability
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, max(y1 - 25, 0)), (x1 + w + 4, max(y1, 25)), (255, 0, 0), -1)
            cv2.putText(image, label, (x1 + 2, max(y1 - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_path, image)
    return detections


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found at {args.model}.")

    print(f"Loading YOLO model from: {args.model}")
    model = YOLO(args.model)

    os.makedirs(args.output, exist_ok=True)

    # Collect source files
    if os.path.isfile(args.source):
        image_files = [args.source]
        source_dir = os.path.dirname(args.source)
    elif os.path.isdir(args.source):
        source_dir = args.source
        image_files = [
            os.path.join(source_dir, f)
            for f in os.listdir(source_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    else:
        raise FileNotFoundError(f"Source path not found: {args.source}")

    print(f"Source: {args.source} ({len(image_files)} image(s))")
    print(f"Output directory: {args.output}")
    print(f"Confidence threshold: {args.conf}\n")

    total_detections = 0
    for idx, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        out_path = os.path.join(args.output, filename)
        detections = process_image(img_path, model, out_path, args.conf)
        total_detections += detections
        print(f"[{idx}/{len(image_files)}] {filename}: {detections} detection(s) -> {out_path}")

    print(f"\nProcessing complete! Processed {len(image_files)} images, detected {total_detections} object(s).")


if __name__ == '__main__':
    main()
