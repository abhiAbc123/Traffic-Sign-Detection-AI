import argparse
import json
import os
import sys
try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
import database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'imageRecognition', 'input_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'imageRecognition', 'DetectionOutput')
MODEL_PATH = os.path.join(BASE_DIR, 'imageRecognition', 'yolov8m.pt')


def get_junction_and_feed_details(junction_query, feed_name, conf=0.25):
    database.init_db()
    junctions = database.get_junctions()

    # Match junction by ID or Name
    matched_junction = None
    if str(junction_query).isdigit():
        jid = int(junction_query)
        matched_junction = next((j for j in junctions if j['id'] == jid), None)
    if not matched_junction:
        q_lower = str(junction_query).lower()
        matched_junction = next((j for j in junctions if q_lower in j['name'].lower() or q_lower in j['location'].lower()), None)

    if not matched_junction:
        matched_junction = junctions[0] if junctions else {
            'id': 1, 'name': 'Default Junction', 'location': 'Main Intersection', 'total_lanes': 4, 'speed_limit': 50
        }

    # Resolve feed image path
    feed_path = None
    if os.path.isabs(feed_name) and os.path.exists(feed_name):
        feed_path = feed_name
    elif os.path.exists(os.path.join(INPUT_DIR, feed_name)):
        feed_path = os.path.join(INPUT_DIR, feed_name)
    elif os.path.exists(os.path.join(OUTPUT_DIR, feed_name)):
        feed_path = os.path.join(OUTPUT_DIR, feed_name)
    elif os.path.exists(feed_name):
        feed_path = os.path.abspath(feed_name)

    if not feed_path:
        # Fallback to first available sample
        samples = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if samples:
            feed_path = os.path.join(INPUT_DIR, samples[0])
            feed_name = samples[0]
        else:
            raise FileNotFoundError(f"Image feed '{feed_name}' not found.")

    filename = os.path.basename(feed_path)
    img = cv2.imread(feed_path)
    if img is None:
        raise ValueError(f"Could not read image: {feed_path}")

    h, w = img.shape[:2]

    # Load Model & Detect
    model = YOLO(MODEL_PATH)
    results = model(img, conf=conf, verbose=False)

    detections = []
    annotated = img.copy()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            c_name = model.names.get(cls_id, str(cls_id))

            detections.append({
                "class_id": cls_id,
                "class_name": c_name,
                "confidence": round(conf_score, 4),
                "box": [x1, y1, x2, y2]
            })

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{c_name} {conf_score:.2f}"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save output annotated image
    out_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")
    cv2.imwrite(out_path, annotated)

    # Record and calculate signal timings in SQLite DB
    timing = database.record_detection_and_timing(
        image_name=filename,
        detections=detections,
        inference_ms=120.0,
        junction_id=matched_junction['id']
    )

    # Vehicle breakdown
    vehicles = {
        "cars": timing['cars_count'],
        "buses": timing['buses_count'],
        "trucks": timing['trucks_count'],
        "motorcycles": timing['motorcycles_count'],
        "pedestrians": timing['pedestrians_count'],
        "total_vehicles": timing['total_vehicles']
    }

    signs_detected = [d['class_name'] for d in detections if d['class_name'].lower() not in ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'person']]

    return {
        "junction": matched_junction,
        "input_feed": {
            "filename": filename,
            "original_photo_path": feed_path,
            "resolution": f"{w}x{h}",
            "annotated_photo_path": out_path
        },
        "signal_details": {
            "signal_state": timing['signal_state'],
            "green_duration_sec": timing['green_duration_sec'],
            "yellow_duration_sec": timing['yellow_duration_sec'],
            "red_duration_sec": timing['red_duration_sec'],
            "cycle_time_sec": timing['cycle_time_sec'],
            "density_score_pcu": timing['density_score'],
            "congestion_level": timing['congestion_level'],
            "queue_clearance_sec": timing['estimated_clearance_sec'],
            "recommended_speed_kmh": timing['recommended_speed_kmh']
        },
        "vehicle_breakdown": vehicles,
        "traffic_signs_detected": signs_detected,
        "detections_count": len(detections)
    }


def main():
    parser = argparse.ArgumentParser(description="Query Monitoring Junction & Input Feed for Signal Details and Photos")
    parser.add_argument('--junction', type=str, default='1', help='Junction ID or Name (e.g. 1, "Central Boulevard", "Highway 101")')
    parser.add_argument('--feed', type=str, default='IMG_5257.jpeg', help='Input image filename or path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')

    args = parser.parse_args()

    details = get_junction_and_feed_details(args.junction, args.feed, args.conf)

    if args.json:
        print(json.dumps(details, indent=2))
        return

    j = details['junction']
    f = details['input_feed']
    s = details['signal_details']
    v = details['vehicle_breakdown']

    print("\n" + "=" * 65)
    print("🚦 SMART TRAFFIC JUNCTION & SIGNAL INSPECTION REPORT")
    print("=" * 65)
    print(f"📍 JUNCTION INFO:")
    print(f"   Name:         {j['name']}")
    print(f"   Location:     {j['location']}")
    print(f"   Lanes:        {j['total_lanes']} lanes | Speed Limit: {j['speed_limit']} km/h")

    print(f"\n📷 FEED & ORIGINAL PHOTO:")
    print(f"   Source Feed:  {f['filename']} ({f['resolution']})")
    print(f"   Original:     {f['original_photo_path']}")
    print(f"   Annotated:    {f['annotated_photo_path']}")

    print(f"\n⏱️ DYNAMIC TRAFFIC SIGNAL TIMINGS:")
    print(f"   Current State:          {s['signal_state']} 🟢")
    print(f"   🟢 Green Duration:      {s['green_duration_sec']} seconds")
    print(f"   🟡 Yellow Clearance:    {s['yellow_duration_sec']} seconds")
    print(f"   🔴 Red Phase Duration:  {s['red_duration_sec']} seconds")
    print(f"   ⏱️ Total Cycle Time:     {s['cycle_time_sec']} seconds")
    print(f"   🚦 Congestion Level:    {s['congestion_level']}")
    print(f"   📊 Traffic Density:     {s['density_score_pcu']} PCU (Passenger Car Units)")
    print(f"   ⏳ Queue Clearance:     {s['queue_clearance_sec']} seconds")
    print(f"   🧭 Advisory Approach:   {s['recommended_speed_kmh']} km/h")

    print(f"\n🚗 VEHICLE & OBJECT BREAKDOWN:")
    print(f"   Total Vehicles:         {v['total_vehicles']} (Cars: {v['cars']}, Buses: {v['buses']}, Trucks: {v['trucks']}, Motorcycles: {v['motorcycles']})")
    print(f"   Pedestrians:            {v['pedestrians']}")
    print(f"   Traffic Signs Found:    {details['traffic_signs_detected'] or 'None'}")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()
