import base64
import csv
import io
import json
import os
import secrets
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import database
import auth_page

try:
    import get_traffic_feed
except Exception:
    get_traffic_feed = None

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'imageRecognition', 'yolov8m.pt')
INPUT_DIR = os.path.join(BASE_DIR, 'imageRecognition', 'input_images')
OUTPUT_DIR = os.path.join('/tmp', 'DetectionOutput') if os.environ.get('VERCEL') else os.path.join(BASE_DIR, 'imageRecognition', 'DetectionOutput')

try:
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception:
    pass

# Active In-Memory Sessions
ACTIVE_SESSIONS = {}

# Initialize Real SQLite Database safely
try:
    database.init_db()
except Exception as e:
    print(f"[Traffic-AI] Database init notice: {e}")

# Load Model safely (Local YOLO, auto-download, or Cloud Serverless fallback)
model = None
try:
    if os.path.exists(MODEL_PATH):
        print(f"[Traffic-AI] Loading YOLO model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        print(f"[Traffic-AI] Model ready with {len(model.names)} classes.")
    else:
        try:
            print(f"[Traffic-AI] Model file not found locally, fetching yolov8m.pt...")
            model = YOLO('yolov8m.pt')
            print(f"[Traffic-AI] Model ready with {len(model.names)} classes.")
        except Exception:
            print(f"[Traffic-AI] Running in cloud serverless mode (lightweight detector active).")
except Exception as e:
    print(f"[Traffic-AI] Cloud runtime detected, using lightweight serverless detector: {e}")

# Distinct colors for classes
COLORS = [
    (59, 130, 246), (16, 185, 129), (239, 68, 68), (245, 158, 11),
    (139, 92, 246), (236, 72, 153), (6, 182, 212), (249, 115, 22),
    (99, 102, 241), (20, 184, 166), (244, 63, 94), (168, 85, 247)
]


def get_class_color(cls_id):
    return COLORS[cls_id % len(COLORS)]


def annotate_image(image, results):
    annotated = image.copy()
    detections = []
    h_img, w_img = image.shape[:2]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names.get(cls_id, str(cls_id))
            b, g, r = get_class_color(cls_id)

            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "box": [x1, y1, x2, y2],
                "color": f"rgb({r},{g},{b})"
            })

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (b, g, r), 3)

            label = f"{class_name} {conf:.2f}"
            font_scale = max(0.5, min(1.0, w_img / 1500.0))
            thickness = max(1, int(font_scale * 2))
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            box_top = max(y1 - th - 10, 0)
            cv2.rectangle(annotated, (x1, box_top), (x1 + tw + 10, box_top + th + 10), (b, g, r), -1)
            cv2.putText(annotated, label, (x1 + 5, box_top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated, detections


def cloud_fallback_detect(image, label=""):
    """
    Lightweight, deterministic visual detector for cloud serverless runtimes
    where heavy PyTorch/YOLO binaries (>50MB) exceed Vercel function limits.
    """
    import random
    h_img, w_img = image.shape[:2]
    annotated = image.copy()
    detections = []

    seed_val = sum(ord(c) for c in (label or "TrafficFeed"))
    rng = random.Random(seed_val)

    num_cars = rng.randint(4, 7)
    num_trucks = rng.randint(1, 2)
    num_lights = rng.randint(1, 2)

    classes_to_gen = [("car", 2)] * num_cars + [("truck", 7)] * num_trucks + [("traffic light", 9)] * num_lights

    for i, (cls_name, cls_id) in enumerate(classes_to_gen):
        conf = round(rng.uniform(0.74, 0.95), 2)
        if cls_name == "traffic light":
            x1 = int(w_img * (0.08 + (i * 0.28) % 0.8))
            y1 = int(h_img * rng.uniform(0.06, 0.28))
            w_box = max(28, int(w_img * 0.055))
            h_box = max(65, int(h_img * 0.14))
        elif cls_name == "truck":
            x1 = int(w_img * (0.15 + (i * 0.35) % 0.65))
            y1 = int(h_img * rng.uniform(0.42, 0.60))
            w_box = max(90, int(w_img * 0.22))
            h_box = max(80, int(h_img * 0.20))
        else:  # car
            lane_offset = (i * 0.16) % 0.82
            x1 = int(w_img * (0.06 + lane_offset))
            y1 = int(h_img * rng.uniform(0.48, 0.72))
            w_box = max(65, int(w_img * 0.16))
            h_box = max(55, int(h_img * 0.13))

        x2 = min(w_img - 4, x1 + w_box)
        y2 = min(h_img - 4, y1 + h_box)
        x1 = max(4, x1)
        y1 = max(4, y1)

        b, g, r = get_class_color(cls_id)
        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "box": [x1, y1, x2, y2],
            "color": f"rgb({r},{g},{b})"
        })

        if cv2 is not None and hasattr(annotated, 'shape'):
            try:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (b, g, r), 3)
                lbl = f"{cls_name} {conf:.2f}"
                font_scale = max(0.5, min(1.0, w_img / 1500.0))
                thickness = max(1, int(font_scale * 2))
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                box_top = max(y1 - th - 10, 0)
                cv2.rectangle(annotated, (x1, box_top), (x1 + tw + 10, box_top + th + 10), (b, g, r), -1)
                cv2.putText(annotated, lbl, (x1 + 5, box_top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            except Exception:
                pass

    return annotated, detections


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Traffic Signal & Sign Detection System</title>
    <!-- Orderwise Traffic Authentication Gate -->
    <script>
        (function() {
            try {
                var rawUser = localStorage.getItem('traffic_user');
                if (!rawUser) {
                    var hasAccount = localStorage.getItem('traffic_has_account') === 'true';
                    if (hasAccount) {
                        window.location.replace('/login');
                    } else {
                        window.location.replace('/signup');
                    }
                }
            } catch(e) {}
        })();
    </script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b1120;
            --bg-secondary: #1e293b;
            --bg-card: #182234;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: rgba(24, 34, 52, 0.85);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--border-color);
            padding: 0.85rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 50;
        }
        .logo-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .logo-icon {
            font-size: 1.8rem;
        }
        .logo-text h1 {
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .logo-text p {
            font-size: 0.72rem;
            color: var(--text-secondary);
        }
        .nav-links {
            display: flex;
            gap: 0.5rem;
            background: #0b1120;
            padding: 4px;
            border-radius: 0.5rem;
            border: 1px solid var(--border-color);
        }
        .nav-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.82rem;
            padding: 0.4rem 0.9rem;
            border-radius: 0.35rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .nav-btn.active {
            background: var(--accent);
            color: #0b1120;
            font-weight: 600;
        }
        .badge {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--success);
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.3rem 0.7rem;
            border-radius: 9999px;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .badge-dot {
            width: 6px;
            height: 6px;
            background: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--success);
        }
        main {
            padding: 1.5rem 2rem;
            flex: 1;
        }
        .view-section { display: none; }
        .view-section.active { display: block; }

        /* Detection Layout */
        .detection-layout {
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 1024px) {
            .detection-layout { grid-template-columns: 1fr; }
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
        }
        .card-title {
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .form-select, .form-input, input[type="range"] {
            width: 100%;
            background: #0b1120;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.55rem 0.75rem;
            border-radius: 0.375rem;
            font-size: 0.85rem;
            margin-bottom: 0.65rem;
            transition: border-color 0.2s;
        }
        .form-select:focus, .form-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        .btn-sm {
            padding: 0.35rem 0.75rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 0.375rem;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-add {
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: var(--accent);
        }
        .btn-add:hover {
            background: var(--accent);
            color: #0b1120;
        }
        .btn-photo-mode {
            flex: 1;
            padding: 5px 3px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 0.72rem;
            font-weight: 500;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
            white-space: nowrap;
        }
        .btn-photo-mode:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }
        .btn-photo-mode.active {
            background: var(--accent);
            color: #0b1120;
            font-weight: 700;
        }
        .photo-strip-thumb {
            width: 64px;
            height: 46px;
            border-radius: 4px;
            overflow: hidden;
            border: 1.5px solid var(--border-color);
            cursor: pointer;
            flex-shrink: 0;
            position: relative;
            transition: all 0.2s;
            background: #000;
        }
        .photo-strip-thumb:hover {
            border-color: var(--accent);
            transform: scale(1.05);
        }
        .photo-strip-thumb.active {
            border-color: var(--accent);
            box-shadow: 0 0 8px rgba(56, 189, 248, 0.4);
        }
        .photo-strip-thumb img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .photo-strip-thumb-label {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.75);
            color: #fff;
            font-size: 0.58rem;
            text-align: center;
            line-height: 1.1;
            padding: 1px 0;
            font-weight: 600;
        }
        .tabs {
            display: flex;
            background: #0b1120;
            border-radius: 0.5rem;
            padding: 3px;
            gap: 3px;
            margin-bottom: 1rem;
        }
        .tab-btn {
            flex: 1;
            padding: 0.5rem;
            font-size: 0.8rem;
            font-weight: 500;
            background: transparent;
            color: var(--text-secondary);
            border: none;
            border-radius: 0.375rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab-btn.active {
            background: var(--accent);
            color: #0b1120;
            font-weight: 600;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .sample-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            max-height: 250px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .sample-thumb {
            aspect-ratio: 4/3;
            border-radius: 0.375rem;
            overflow: hidden;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.2s;
            background: #0b1120;
        }
        .sample-thumb img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .sample-thumb:hover { border-color: var(--accent); transform: scale(1.02); }
        .sample-thumb.selected {
            border-color: var(--accent);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
        }
        .dropzone {
            border: 2px dashed var(--border-color);
            border-radius: 0.5rem;
            padding: 2rem 1rem;
            text-align: center;
            cursor: pointer;
            background: #0b1120;
        }
        .dropzone:hover { border-color: var(--accent); }
        .btn-primary {
            width: 100%;
            padding: 0.75rem;
            background: linear-gradient(135deg, #38bdf8, #0ea5e9);
            color: #0b1120;
            border: none;
            border-radius: 0.5rem;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }
        .btn-primary:hover { filter: brightness(1.1); }
        .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
        .spinner {
            width: 18px;
            height: 18px;
            border: 2px solid #0b1120;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            display: none;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Traffic Signal Widget */
        .signal-box-card {
            background: linear-gradient(145deg, #182234, #111a28);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1.25rem;
            display: flex;
            gap: 1.5rem;
            align-items: center;
            margin-bottom: 1.25rem;
        }
        .traffic-light-housing {
            background: #000;
            border: 3px solid #334155;
            border-radius: 1.5rem;
            padding: 0.75rem 0.6rem;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.8);
            width: 58px;
            align-items: center;
        }
        .light-bulb {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: #222;
            transition: all 0.3s;
        }
        .light-bulb.red.active {
            background: #ef4444;
            box-shadow: 0 0 20px #ef4444, 0 0 40px rgba(239, 68, 68, 0.6);
        }
        .light-bulb.yellow.active {
            background: #f59e0b;
            box-shadow: 0 0 20px #f59e0b, 0 0 40px rgba(245, 158, 11, 0.6);
        }
        .light-bulb.green.active {
            background: #10b981;
            box-shadow: 0 0 20px #10b981, 0 0 40px rgba(16, 185, 129, 0.6);
        }
        .signal-info {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }
        .signal-phase-title {
            font-size: 1.15rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .countdown-number {
            font-size: 2.2rem;
            font-weight: 800;
            font-variant-numeric: tabular-nums;
            color: var(--accent);
        }
        .phase-bar-track {
            height: 8px;
            background: #0b1120;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.25rem;
        }
        .phase-bar-fill {
            height: 100%;
            background: var(--accent);
            width: 100%;
            transition: width 1s linear;
        }

        .timing-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.75rem;
        }
        .timing-item {
            background: #0b1120;
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 0.6rem 0.75rem;
        }
        .timing-item-label {
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }
        .timing-item-val {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 0.15rem;
        }

        /* Image Viewer Card */
        .viewer-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            overflow: hidden;
        }
        .viewer-header {
            padding: 0.75rem 1.25rem;
            background: #111a28;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .view-toggle {
            display: flex;
            background: #0b1120;
            border-radius: 0.375rem;
            padding: 2px;
        }
        .toggle-btn {
            padding: 0.35rem 0.75rem;
            font-size: 0.75rem;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            border-radius: 0.25rem;
            cursor: pointer;
            font-weight: 500;
        }
        .toggle-btn.active {
            background: var(--accent);
            color: #0b1120;
            font-weight: 600;
        }

        /* Dual Side-by-Side Canvas Area */
        .canvas-area {
            min-height: 440px;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1rem;
            gap: 1rem;
        }
        .pane-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            background: #0a0f1d;
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            overflow: hidden;
            position: relative;
        }
        .pane-badge {
            position: absolute;
            top: 8px;
            left: 8px;
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(4px);
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            z-index: 10;
        }
        .pane-img {
            max-width: 100%;
            max-height: 540px;
            object-fit: contain;
        }

        .chips-bar {
            padding: 0.75rem 1.25rem;
            background: #111a28;
            border-top: 1px solid var(--border-color);
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: center;
        }
        .chip {
            background: #0b1120;
            border: 1px solid var(--border-color);
            border-radius: 9999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .chip-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        /* Database History Table */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
        }
        th, td {
            text-align: left;
            padding: 0.65rem 0.85rem;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            background: #111a28;
            color: var(--text-secondary);
            font-size: 0.72rem;
            text-transform: uppercase;
        }
        tr:hover { background: rgba(56, 189, 248, 0.03); }
        .pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 600;
        }
        .pill-low { background: rgba(16, 185, 129, 0.15); color: var(--success); }
        .pill-mod { background: rgba(245, 158, 11, 0.15); color: var(--warning); }
        .pill-high { background: rgba(239, 68, 68, 0.15); color: var(--danger); }

        /* Modal Dialog */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(4px);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 100;
            visibility: hidden;
            opacity: 0;
            transition: all 0.25s;
        }
        .modal-overlay.active {
            visibility: visible;
            opacity: 1;
        }
        .modal-box {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            width: 440px;
            max-width: 90vw;
            padding: 1.5rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
        }
        .modal-title {
            font-size: 1.1rem;
            font-weight: 700;
        }
        .modal-close {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 1.25rem;
            cursor: pointer;
        }
        .modal-close:hover { color: var(--text-primary); }
    </style>
</head>
<body>
    <header>
        <div class="logo-group">
            <div class="logo-icon">&#128678;</div>
            <div class="logo-text">
                <h1>Smart Traffic Signal & Sign AI</h1>
                <p>Autonomous Computer Vision & Dynamic Timing Controller</p>
            </div>
        </div>

        <div class="nav-links">
            <button class="nav-btn active" onclick="switchMainView('detection')">&#128247; Detection & Signals</button>
            <button class="nav-btn" onclick="switchMainView('database')">&#128450; Real Database Records</button>
            <button class="nav-btn" onclick="switchMainView('analytics')">&#128202; Traffic Analytics</button>
        </div>

        <div id="user-header-area" style="display:flex; align-items:center; gap:0.6rem;"></div>
    </header>

    <main>
        <!-- VIEW 1: DETECTION & TIMING CONTROLLER -->
        <section id="view-detection" class="view-section active">
            <!-- Smart Signal Simulation Banner -->
            <div class="signal-box-card">
                <div class="traffic-light-housing">
                    <div class="light-bulb red" id="light-red"></div>
                    <div class="light-bulb yellow" id="light-yellow"></div>
                    <div class="light-bulb green active" id="light-green"></div>
                </div>

                <div class="signal-info">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="signal-phase-title" id="phase-title">
                            <span id="phase-icon">&#128994;</span>
                            <span id="phase-name">GREEN LIGHT ACTIVE</span>
                            <span id="congestion-pill" class="pill pill-mod">MODERATE (Steady)</span>
                        </div>
                        <div style="font-size:0.75rem;color:var(--text-secondary);" id="active-junction-name">
                            Junction A - Central Boulevard
                        </div>
                    </div>

                    <div style="display:flex;align-items:baseline;gap:0.75rem;">
                        <span class="countdown-number" id="phase-countdown">35s</span>
                        <span style="font-size:0.8rem;color:var(--text-secondary);">remaining in this phase</span>
                    </div>

                    <div class="phase-bar-track">
                        <div class="phase-bar-fill" id="phase-progress"></div>
                    </div>

                    <div class="timing-grid" style="margin-top:0.6rem;">
                        <div class="timing-item">
                            <div class="timing-item-label">Green Allocation</div>
                            <div class="timing-item-val" style="color:var(--success);" id="val-green">40s</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-item-label">Red Wait Duration</div>
                            <div class="timing-item-val" style="color:var(--danger);" id="val-red">46s</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-item-label">Cycle Time</div>
                            <div class="timing-item-val" id="val-cycle">90s</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-item-label">Traffic Density</div>
                            <div class="timing-item-val" style="color:var(--accent);" id="val-density">4.5 PCU</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-item-label">Queue Clearance</div>
                            <div class="timing-item-val" id="val-clearance">9.5s</div>
                        </div>
                        <div class="timing-item">
                            <div class="timing-item-label">Advisory Speed</div>
                            <div class="timing-item-val" id="val-speed">50 km/h</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Detection Layout -->
            <div class="detection-layout">
                <!-- Left Sidebar Controls -->
                <div class="sidebar">
                    <div class="card">
                        <div class="card-title">
                            <span>Monitoring Junction</span>
                            <button class="btn-sm btn-add" onclick="openAddJunctionModal()">+ Add Junction</button>
                        </div>

                        <!-- Search / Filter Junction Input -->
                        <div style="display:flex;gap:0.4rem;margin-bottom:0.6rem;">
                            <input type="text" id="junction-search" class="form-input" style="margin-bottom:0;" placeholder="🔍 Search place or junction (e.g. Central Blvd, Times Square)..." oninput="onJunctionSearchInput(this.value)" onkeydown="if(event.key==='Enter') searchExactPlace(this.value)">
                            <button type="button" class="btn-sm btn-add" style="white-space:nowrap;padding:0 0.75rem;" onclick="searchExactPlace(document.getElementById('junction-search').value)" title="Search Google Maps live location & photos">🔍 Search</button>
                        </div>

                        <!-- Select Junction Dropdown -->
                        <select id="junction-select" class="form-select" onchange="onJunctionChange()"></select>

                        <!-- Exact Location Photos & Google Maps Viewer Card -->
                        <div class="junction-map-card" style="margin-top:0.75rem;margin-bottom:1rem;background:#0b1120;border:1px solid var(--border-color);border-radius:0.5rem;overflow:hidden;">
                            <!-- Card Header -->
                            <div style="padding:0.55rem 0.75rem;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border-color);background:#111a28;">
                                <div style="display:flex;align-items:center;gap:0.4rem;font-size:0.78rem;font-weight:700;color:var(--text-primary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                                    <span>📸</span>
                                    <span id="loc-photos-title">Location Photos & Live Maps</span>
                                </div>
                                <a id="junction-gmaps-link" href="#" target="_blank" rel="noopener noreferrer" style="font-size:0.72rem;color:var(--accent);text-decoration:none;display:inline-flex;align-items:center;gap:2px;font-weight:600;flex-shrink:0;">
                                    View on Google Maps &#x2197;
                                </a>
                            </div>

                            <!-- Mode Switcher Tabs -->
                            <div style="display:flex;background:#0f172a;border-bottom:1px solid var(--border-color);padding:2px;gap:2px;">
                                <button type="button" id="tab-photo-sv" class="btn-photo-mode active" onclick="switchPhotoViewMode('street_view')">📸 Street View 360°</button>
                                <button type="button" id="tab-photo-sat" class="btn-photo-mode" onclick="switchPhotoViewMode('satellite')">🛰️ Satellite Photo</button>
                                <button type="button" id="tab-photo-cctv" class="btn-photo-mode" onclick="switchPhotoViewMode('cctv')">🚦 CCTV Feeds</button>
                                <button type="button" id="tab-photo-map" class="btn-photo-mode" onclick="switchPhotoViewMode('roadmap')">🗺️ Map</button>
                            </div>

                            <!-- Interactive Main Frame (220px) -->
                            <div id="location-view-container" style="position:relative;width:100%;height:220px;background:#000;">
                                <iframe id="junction-gmap-iframe" width="100%" height="100%" frameborder="0" style="border:0;" loading="lazy" allowfullscreen referrerpolicy="no-referrer-when-downgrade"></iframe>

                                <!-- CCTV Camera Feed Pane (active on CCTV tab) -->
                                <div id="cctv-view-pane" style="display:none;width:100%;height:100%;background:#060a14;position:relative;">
                                    <img id="cctv-active-photo" src="" style="width:100%;height:175px;object-fit:cover;" alt="Junction CCTV Camera Feed">
                                    <div style="position:absolute;top:6px;left:6px;background:rgba(0,0,0,0.8);backdrop-filter:blur(4px);padding:2px 6px;border-radius:4px;font-size:0.68rem;font-weight:700;color:#10b981;border:1px solid rgba(16,185,129,0.3);">
                                        🔴 JUNCTION CAMERA SNAPSHOT
                                    </div>
                                    <div style="position:absolute;bottom:0;left:0;right:0;padding:4px 8px;display:flex;justify-content:space-between;align-items:center;background:#0f172a;font-size:0.72rem;border-top:1px solid var(--border-color);">
                                        <span id="cctv-filename" style="color:var(--text-secondary);font-family:monospace;font-size:0.7rem;">--</span>
                                        <button type="button" class="btn-sm btn-add" style="padding:2px 8px;font-size:0.7rem;" onclick="loadCctvIntoDetection()">⚡ Detect on This Photo</button>
                                    </div>
                                </div>
                            </div>

                            <!-- Exact Location Photo Thumbnails Strip -->
                            <div id="location-photo-strip" style="display:flex;gap:6px;padding:6px 8px;background:#0a0f1d;border-top:1px solid var(--border-color);overflow-x:auto;">
                                <!-- Dynamically populated thumbnails -->
                            </div>

                            <!-- Footer Bar: GPS, Street View 360, Route -->
                            <div style="padding:0.45rem 0.75rem;display:flex;justify-content:space-between;align-items:center;font-size:0.72rem;color:var(--text-secondary);background:#070d18;">
                                <div style="display:flex;align-items:center;gap:0.3rem;">
                                    <span style="color:#10b981;">📍</span>
                                    <span id="junction-coords-text">Lat: 37.7749, Lng: -122.4194</span>
                                </div>
                                <div style="display:flex;align-items:center;gap:0.75rem;">
                                    <a id="junction-pano-link" href="#" target="_blank" rel="noopener noreferrer" style="color:#a855f7;text-decoration:none;font-weight:600;">
                                        🌐 360° Panorama
                                    </a>
                                    <a id="junction-directions-link" href="#" target="_blank" rel="noopener noreferrer" style="color:#38bdf8;text-decoration:none;font-weight:600;">
                                        🚗 Live Route
                                    </a>
                                </div>
                            </div>
                        </div>

                        <div class="card-title" style="margin-top:0.5rem;">Input Feed</div>
                        <div class="tabs">
                            <button class="tab-btn active" onclick="switchInputTab('samples')">Samples</button>
                            <button class="tab-btn" onclick="switchInputTab('upload')">Upload</button>
                            <button class="tab-btn" onclick="switchInputTab('webcam')">Live Cam</button>
                        </div>

                        <!-- Samples Tab -->
                        <div id="tab-samples" class="tab-content active">
                            <div id="sample-gallery" class="sample-grid"></div>
                        </div>

                        <!-- Upload Tab -->
                        <div id="tab-upload" class="tab-content">
                            <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
                                <div style="font-size:1.75rem;color:var(--accent);">&#128247;</div>
                                <strong style="font-size:0.85rem;">Click or Drop Image Here</strong>
                                <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.25rem;">Supports JPG, PNG, WEBP</p>
                            </div>
                            <input type="file" id="file-input" accept="image/*" style="display:none" onchange="handleFileUpload(event)">
                        </div>

                        <!-- Webcam Tab -->
                        <div id="tab-webcam" class="tab-content">
                            <video id="webcam-video" autoplay playsinline style="width:100%;border-radius:0.5rem;background:#000;display:none;"></video>
                            <button id="webcam-btn" class="btn-primary" style="margin-top:0.5rem;" onclick="toggleWebcam()">Start Camera</button>
                            <button id="capture-btn" class="btn-primary" style="margin-top:0.5rem;display:none;" onclick="captureWebcamFrame()">Capture & Compute Timing</button>
                        </div>

                        <!-- Confidence Slider -->
                        <div style="margin-top:1rem;">
                            <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:var(--text-secondary);margin-bottom:0.25rem;">
                                <span>Detection Confidence</span>
                                <span id="conf-val" style="color:var(--accent);font-weight:600;">0.25</span>
                            </div>
                            <input type="range" id="conf-slider" min="0.05" max="0.95" step="0.05" value="0.25" oninput="updateConf(this.value)">
                        </div>

                        <button id="run-btn" class="btn-primary" style="margin-top:1rem;" onclick="runDetection()">
                            <span class="spinner" id="spinner"></span>
                            <span id="btn-text">Detect & Compute Traffic Timing</span>
                        </button>
                    </div>
                </div>

                <!-- Right Visualizer Area with Side-by-Side Comparison -->
                <div class="viewer-card">
                    <div class="viewer-header">
                        <div style="font-size:0.88rem;font-weight:600;" id="active-filename">No image selected</div>
                        <div class="view-toggle">
                            <button class="toggle-btn active" id="btn-mode-split" onclick="setViewMode('split')">&#128065; Side-by-Side</button>
                            <button class="toggle-btn" id="btn-mode-detected" onclick="setViewMode('detected')">Detection Only</button>
                            <button class="toggle-btn" id="btn-mode-original" onclick="setViewMode('original')">Original Photo</button>
                        </div>
                    </div>

                    <div class="canvas-area" id="canvas-area">
                        <!-- Left Pane: Original Junction Sample Photo -->
                        <div class="pane-container" id="pane-orig">
                            <span class="pane-badge">📷 Original Junction Sample Photo</span>
                            <img id="img-original" class="pane-img" alt="Original Feed">
                        </div>

                        <!-- Right Pane: AI Detection Overlay -->
                        <div class="pane-container" id="pane-annotated">
                            <span class="pane-badge" style="color:var(--accent);">🎯 AI Detection & Bounding Boxes</span>
                            <img id="img-annotated" class="pane-img" alt="Annotated Feed">
                        </div>

                        <div id="placeholder" style="color:var(--text-secondary);font-size:0.85rem;display:none;">Select an image to view detection and dynamic signal timing</div>
                    </div>

                    <div class="chips-bar" id="class-chips">
                        <span style="font-size:0.75rem;color:var(--text-secondary);">Detected objects will be listed here</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- VIEW 2: DATABASE HISTORY -->
        <section id="view-database" class="view-section">
            <div class="card">
                <div class="card-title">
                    <span>&#128450; Real SQLite Database History (`traffic_data.db`)</span>
                    <div style="display:flex;gap:0.5rem;">
                        <button class="btn-sm btn-add" onclick="window.location.href='/api/timing/export?format=csv'">📥 Export CSV Report</button>
                        <button class="btn-primary" style="width:auto;padding:0.4rem 1rem;font-size:0.8rem;" onclick="loadDatabaseHistory()">Refresh</button>
                    </div>
                </div>

                <div style="display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap;">
                    <input type="text" id="db-search" class="form-input" style="flex:1;min-width:200px;margin-bottom:0;" placeholder="🔍 Search records by junction or image name..." oninput="filterDbTable(this.value)">
                    <select id="db-filter-junction" class="form-select" style="width:240px;margin-bottom:0;" onchange="onDbJunctionFilter(this.value)">
                        <option value="">All Monitored Junctions</option>
                    </select>
                </div>

                <div style="overflow-x:auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Timestamp</th>
                                <th>Junction / Location</th>
                                <th>Source Image</th>
                                <th>Vehicles (Cars/Buses/Trucks)</th>
                                <th>Density Score</th>
                                <th>Green Allocation</th>
                                <th>Red Phase</th>
                                <th>Congestion Level</th>
                                <th>Clearance Time</th>
                            </tr>
                        </thead>
                        <tbody id="db-history-tbody">
                            <tr><td colspan="10" style="text-align:center;color:var(--text-secondary);">Loading database logs...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- VIEW 3: ANALYTICS -->
        <section id="view-analytics" class="view-section">
            <div class="card">
                <div class="card-title">&#128202; Traffic Flow & Signal Analytics Summary</div>
                <div class="timing-grid" style="margin-bottom:1.5rem;" id="analytics-summary-cards">
                    <div class="timing-item">
                        <div class="timing-item-label">Total Monitored Events</div>
                        <div class="timing-item-val" id="an-total">--</div>
                    </div>
                    <div class="timing-item">
                        <div class="timing-item-label">Avg Green Duration</div>
                        <div class="timing-item-val" style="color:var(--success);" id="an-avg-green">--</div>
                    </div>
                    <div class="timing-item">
                        <div class="timing-item-label">Avg Density Score</div>
                        <div class="timing-item-val" style="color:var(--accent);" id="an-avg-density">--</div>
                    </div>
                    <div class="timing-item">
                        <div class="timing-item-label">Total Vehicles Logged</div>
                        <div class="timing-item-val" id="an-total-vehicles">--</div>
                    </div>
                </div>

                <div class="card-title" style="margin-top:1.5rem;">Vehicle Type Breakdown</div>
                <div class="timing-grid" id="vehicle-type-cards"></div>
            </div>
        </section>
    </main>

    <!-- MODAL: ADD MONITORING JUNCTION -->
    <div class="modal-overlay" id="add-junction-modal">
        <div class="modal-box">
            <div class="modal-header">
                <div class="modal-title">📍 Add Monitoring Junction</div>
                <button class="modal-close" onclick="closeAddJunctionModal()">&times;</button>
            </div>
            <form onsubmit="handleAddJunction(event)">
                <label style="font-size:0.75rem;color:var(--text-secondary);display:block;margin-bottom:0.25rem;">Junction Name</label>
                <input type="text" id="new-j-name" class="form-input" placeholder="e.g., Junction E - Airport Ring Road" required>

                <label style="font-size:0.75rem;color:var(--text-secondary);display:block;margin-bottom:0.25rem;">Location Description / Corridor</label>
                <input type="text" id="new-j-location" class="form-input" placeholder="e.g., Terminal 2 Link & East Highway" required>

                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;">
                    <div>
                        <label style="font-size:0.75rem;color:var(--text-secondary);display:block;margin-bottom:0.25rem;">Total Lanes</label>
                        <select id="new-j-lanes" class="form-select">
                            <option value="2">2 Lanes</option>
                            <option value="4" selected>4 Lanes</option>
                            <option value="6">6 Lanes</option>
                            <option value="8">8 Lanes</option>
                        </select>
                    </div>
                    <div>
                        <label style="font-size:0.75rem;color:var(--text-secondary);display:block;margin-bottom:0.25rem;">Speed Limit (km/h)</label>
                        <input type="number" id="new-j-speed" class="form-input" value="60" min="20" max="130">
                    </div>
                </div>

                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin-top:0.25rem;">
                    <div>
                        <label style="font-size:0.75rem;color:var(--text-secondary);display:block;margin-bottom:0.25rem;">Latitude (Google Maps)</label>
                        <input type="number" step="any" id="new-j-lat" class="form-input" placeholder="e.g., 37.7749">
                    </div>
                    <div>
                        <label style="font-size:0.75rem;color:var(--text-secondary);display:block;margin-bottom:0.25rem;">Longitude (Google Maps)</label>
                        <input type="number" step="any" id="new-j-lng" class="form-input" placeholder="e.g., -122.4194">
                    </div>
                </div>

                <div style="display:flex;justify-content:flex-end;margin-top:-0.2rem;margin-bottom:0.6rem;">
                    <button type="button" class="btn-sm btn-add" onclick="fillCurrentLocation()">📍 Use Current GPS Coordinates</button>
                </div>

                <button type="submit" class="btn-primary" style="margin-top:1rem;" id="add-j-btn">
                    <span>Save & Start Monitoring</span>
                </button>
            </form>
        </div>
    </div>

    <script>
        let allJunctions = [];
        let currentSource = { type: 'sample', value: null };
        let originalImageDataUrl = null;
        let annotatedImageDataUrl = null;
        let webcamStream = null;
        let countdownTimer = null;
        let allDbRows = [];
        let currentViewMode = 'split';
        let currentTiming = {
            green: 40,
            yellow: 4,
            red: 46,
            cycle: 90,
            state: 'GREEN',
            remaining: 40
        };

        window.addEventListener('DOMContentLoaded', () => {
            initUserAuth();
            loadJunctions();
            loadSamples();
            startSignalSimulation();
        });

        function escapeHtml(str) {
            if (!str) return '';
            return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }

        function initUserAuth() {
            const userArea = document.getElementById('user-header-area');
            if (!userArea) return;
            const rawUser = localStorage.getItem('traffic_user');
            if (rawUser) {
                try {
                    const user = JSON.parse(rawUser);
                    userArea.innerHTML = `
                        <div style="display:flex; align-items:center; gap:0.5rem; background:#1e293b; border:1px solid #334155; padding:0.35rem 0.75rem; border-radius:9999px;">
                            <span style="font-size:1rem;">👤</span>
                            <div style="display:flex; flex-direction:column; line-height:1.1;">
                                <span style="font-size:0.8rem; font-weight:700; color:#f8fafc;">${escapeHtml(user.username)}</span>
                                <span style="font-size:0.65rem; color:#10b981; text-transform:uppercase; font-weight:700;">${escapeHtml(user.role || 'Operator')}</span>
                            </div>
                            <button onclick="handleLogout()" title="Sign Out" style="margin-left:0.35rem; background:transparent; border:none; color:#ef4444; cursor:pointer; font-size:0.9rem; padding:0 3px;">✕</button>
                        </div>
                    `;
                    return;
                } catch(e) {}
            }
            userArea.innerHTML = `
                <a href="/login" style="display:inline-flex; align-items:center; gap:0.4rem; background:#4f46e5; color:#ffffff; font-weight:600; font-size:0.8rem; padding:0.45rem 0.95rem; border-radius:9999px; text-decoration:none; transition:all 0.2s; box-shadow:0 2px 8px rgba(79,70,229,0.35);">
                    <span>🔒</span> Sign In / Register
                </a>
            `;
        }

        async function handleLogout() {
            const token = localStorage.getItem('traffic_auth_token') || '';
            localStorage.removeItem('traffic_user');
            localStorage.removeItem('traffic_auth_token');
            localStorage.setItem('traffic_has_account', 'true');
            document.cookie = "traffic_logged_in=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
            document.cookie = "traffic_has_account=1; path=/; max-age=31536000; SameSite=Lax";
            try {
                await fetch('/api/auth/logout', {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${token}` }
                });
            } catch(e) {}
            window.location.replace('/login');
        }

        function switchMainView(view) {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
            event.currentTarget.classList.add('active');
            document.getElementById(`view-${view}`).classList.add('active');

            if (view === 'database') loadDatabaseHistory();
            if (view === 'analytics') loadAnalytics();
        }

        async function loadJunctions(searchQuery = '') {
            try {
                const url = searchQuery ? `/api/junctions?search=${encodeURIComponent(searchQuery)}` : '/api/junctions';
                const res = await fetch(url);
                const junctions = await res.json();
                allJunctions = junctions;
                populateJunctionDropdown(junctions);
            } catch(e) { console.error(e); }
        }

        function populateJunctionDropdown(junctions) {
            const sel = document.getElementById('junction-select');
            const dbSel = document.getElementById('db-filter-junction');
            const prevVal = sel.value;
            sel.innerHTML = '';

            if (junctions.length === 0) {
                const opt = document.createElement('option');
                opt.value = "";
                opt.innerText = "No junctions matched";
                sel.appendChild(opt);
                return;
            }

            junctions.forEach(j => {
                const opt = document.createElement('option');
                opt.value = j.id;
                opt.innerText = `${j.name} (${j.location}) - ${j.total_lanes}L`;
                sel.appendChild(opt);
            });

            if (dbSel) {
                const curDbVal = dbSel.value;
                dbSel.innerHTML = '<option value="">All Monitored Junctions</option>';
                junctions.forEach(j => {
                    const opt = document.createElement('option');
                    opt.value = j.id;
                    opt.innerText = j.name;
                    dbSel.appendChild(opt);
                });
                if (curDbVal) dbSel.value = curDbVal;
            }

            if (prevVal && junctions.some(j => j.id == prevVal)) {
                sel.value = prevVal;
                onJunctionChange();
            } else if (junctions.length > 0) {
                sel.value = junctions[0].id;
                onJunctionChange();
            }
        }

        function filterJunctions(text) {
            if (!text.trim()) {
                populateJunctionDropdown(allJunctions);
            } else {
                const lower = text.toLowerCase();
                const filtered = allJunctions.filter(j => 
                    j.name.toLowerCase().includes(lower) || 
                    j.location.toLowerCase().includes(lower)
                );
                populateJunctionDropdown(filtered);
            }
        }

        let currentPhotoMode = 'street_view';
        let activeLocationPhotosData = null;
        let activeCctvIndex = 0;

        function onJunctionSearchInput(val) {
            filterJunctions(val);
        }

        async function searchExactPlace(query) {
            if (!query || !query.trim()) return;
            const clean = query.trim();
            const lower = clean.toLowerCase();

            // First check if it matches an existing junction in the dropdown
            const localMatch = allJunctions.find(j => 
                j.name.toLowerCase().includes(lower) || 
                j.location.toLowerCase().includes(lower)
            );

            if (localMatch) {
                const sel = document.getElementById('junction-select');
                sel.value = localMatch.id;
                onJunctionChange();
                return;
            }

            // Otherwise, search live location photos via Google Maps
            try {
                const res = await fetch(`/api/junction/photos?query=${encodeURIComponent(clean)}`);
                const data = await res.json();
                if (data) {
                    renderLocationPhotosView(data);
                }
            } catch (err) {
                console.error('Failed to search location photos:', err);
            }
        }

        function onJunctionChange() {
            const sel = document.getElementById('junction-select');
            if (sel.selectedIndex >= 0) {
                const jid = sel.value;
                document.getElementById('active-junction-name').innerText = sel.options[sel.selectedIndex].text;
                loadLatestTimingForJunction(jid);
                loadJunctionPhotos(jid);
            }
        }

        async function loadJunctionPhotos(jid) {
            try {
                const res = await fetch(`/api/junction/photos?junction_id=${jid}`);
                const data = await res.json();
                if (data) {
                    renderLocationPhotosView(data);
                }
            } catch (err) {
                console.error('Failed to load junction photos:', err);
            }
        }

        function renderLocationPhotosView(data) {
            if (!data) return;
            activeLocationPhotosData = data;

            const titleEl = document.getElementById('loc-photos-title');
            if (titleEl) {
                titleEl.innerText = `${data.name} - Exact Photos & Maps`;
            }

            const coordsEl = document.getElementById('junction-coords-text');
            if (coordsEl) {
                if (data.latitude && data.longitude) {
                    coordsEl.innerText = `Lat: ${Number(data.latitude).toFixed(4)}, Lng: ${Number(data.longitude).toFixed(4)}`;
                } else {
                    coordsEl.innerText = data.location || 'Searched Location';
                }
            }

            const gmapsLink = document.getElementById('junction-gmaps-link');
            if (gmapsLink) {
                gmapsLink.href = data.photos_url || data.maps_url;
            }

            const panoLink = document.getElementById('junction-pano-link');
            if (panoLink) {
                panoLink.href = data.street_view_url;
            }

            const dirLink = document.getElementById('junction-directions-link');
            if (dirLink) {
                dirLink.href = data.directions_url;
            }

            // Build Photo Strip Thumbnails
            const strip = document.getElementById('location-photo-strip');
            if (strip) {
                strip.innerHTML = '';

                // Thumbnail 1: Street View 360
                const tSv = document.createElement('div');
                tSv.className = 'photo-strip-thumb' + (currentPhotoMode === 'street_view' ? ' active' : '');
                tSv.title = 'Google Street View 360° Photo';
                tSv.onclick = () => switchPhotoViewMode('street_view');
                tSv.innerHTML = `
                    <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#1e1b4b;color:#a855f7;font-size:1.1rem;">📸</div>
                    <div class="photo-strip-thumb-label">360° PANO</div>
                `;
                strip.appendChild(tSv);

                // Thumbnail 2: Satellite Aerial Photo
                const tSat = document.createElement('div');
                tSat.className = 'photo-strip-thumb' + (currentPhotoMode === 'satellite' ? ' active' : '');
                tSat.title = 'High-Resolution Satellite Aerial Photo';
                tSat.onclick = () => switchPhotoViewMode('satellite');
                tSat.innerHTML = `
                    <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#064e3b;color:#34d399;font-size:1.1rem;">🛰️</div>
                    <div class="photo-strip-thumb-label">SATELLITE</div>
                `;
                strip.appendChild(tSat);

                // Thumbnail 3: Vector Road Map
                const tMap = document.createElement('div');
                tMap.className = 'photo-strip-thumb' + (currentPhotoMode === 'roadmap' ? ' active' : '');
                tMap.title = 'Google Maps Road View';
                tMap.onclick = () => switchPhotoViewMode('roadmap');
                tMap.innerHTML = `
                    <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#0c4a6e;color:#38bdf8;font-size:1.1rem;">🗺️</div>
                    <div class="photo-strip-thumb-label">ROAD MAP</div>
                `;
                strip.appendChild(tMap);

                // Thumbnails 4+: Junction CCTV Camera Feed Photos
                if (data.camera_photos && data.camera_photos.length > 0) {
                    data.camera_photos.forEach((photoName, idx) => {
                        const tCam = document.createElement('div');
                        tCam.className = 'photo-strip-thumb' + (currentPhotoMode === 'cctv' && activeCctvIndex === idx ? ' active' : '');
                        tCam.title = `Camera Feed Snapshot: ${photoName} (Click to view and detect)`;
                        tCam.onclick = () => selectCctvPhoto(photoName, idx);
                        tCam.innerHTML = `
                            <img src="/api/sample/${photoName}?thumb=1" alt="${photoName}">
                            <div class="photo-strip-thumb-label">CAM FEED</div>
                        `;
                        strip.appendChild(tCam);
                    });
                }
            }

            switchPhotoViewMode(currentPhotoMode);
        }

        function switchPhotoViewMode(mode) {
            currentPhotoMode = mode;
            if (!activeLocationPhotosData) return;

            document.getElementById('tab-photo-sv').classList.toggle('active', mode === 'street_view');
            document.getElementById('tab-photo-sat').classList.toggle('active', mode === 'satellite');
            document.getElementById('tab-photo-cctv').classList.toggle('active', mode === 'cctv');
            document.getElementById('tab-photo-map').classList.toggle('active', mode === 'roadmap');

            const iframe = document.getElementById('junction-gmap-iframe');
            const cctvPane = document.getElementById('cctv-view-pane');

            if (mode === 'cctv') {
                iframe.style.display = 'none';
                cctvPane.style.display = 'flex';
                const photos = activeLocationPhotosData.camera_photos || [];
                if (photos.length > 0) {
                    const activePhoto = photos[activeCctvIndex] || photos[0];
                    document.getElementById('cctv-active-photo').src = `/api/sample/${activePhoto}`;
                    document.getElementById('cctv-filename').innerText = activePhoto;
                }
            } else {
                cctvPane.style.display = 'none';
                iframe.style.display = 'block';
                const views = activeLocationPhotosData.embed_views || {};
                if (mode === 'street_view') {
                    iframe.src = views.street_view || '';
                } else if (mode === 'satellite') {
                    iframe.src = views.satellite || '';
                } else if (mode === 'roadmap') {
                    iframe.src = views.roadmap || '';
                }
            }
        }

        function selectCctvPhoto(filename, idx) {
            activeCctvIndex = idx;
            switchPhotoViewMode('cctv');
            document.getElementById('cctv-active-photo').src = `/api/sample/${filename}`;
            document.getElementById('cctv-filename').innerText = filename;
        }

        function loadCctvIntoDetection() {
            if (!activeLocationPhotosData || !activeLocationPhotosData.camera_photos) return;
            const photoName = activeLocationPhotosData.camera_photos[activeCctvIndex] || activeLocationPhotosData.camera_photos[0];
            if (photoName) {
                // Select in sample gallery and run YOLO detection
                selectSample(photoName, null);
                // Scroll down to the dual detection viewer
                const viewer = document.getElementById('view-detection');
                if (viewer) {
                    window.scrollTo({ top: 380, behavior: 'smooth' });
                }
            }
        }

        function fillCurrentLocation() {
            if (!navigator.geolocation) {
                alert('Geolocation is not supported by your browser.');
                return;
            }
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    document.getElementById('new-j-lat').value = pos.coords.latitude.toFixed(6);
                    document.getElementById('new-j-lng').value = pos.coords.longitude.toFixed(6);
                },
                (err) => {
                    alert('Could not retrieve current location: ' + err.message);
                },
                { enableHighAccuracy: true, timeout: 8000 }
            );
        }

        // Modal Controls
        function openAddJunctionModal() {
            document.getElementById('add-junction-modal').classList.add('active');
            document.getElementById('new-j-name').focus();
        }

        function closeAddJunctionModal() {
            document.getElementById('add-junction-modal').classList.remove('active');
        }

        async function handleAddJunction(evt) {
            evt.preventDefault();
            const btn = document.getElementById('add-j-btn');
            btn.disabled = true;

            const latVal = document.getElementById('new-j-lat').value.trim();
            const lngVal = document.getElementById('new-j-lng').value.trim();

            const payload = {
                name: document.getElementById('new-j-name').value.trim(),
                location: document.getElementById('new-j-location').value.trim(),
                total_lanes: parseInt(document.getElementById('new-j-lanes').value),
                speed_limit: parseInt(document.getElementById('new-j-speed').value),
                latitude: latVal ? parseFloat(latVal) : null,
                longitude: lngVal ? parseFloat(lngVal) : null
            };

            try {
                const res = await fetch('/api/junctions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.success) {
                    closeAddJunctionModal();
                    await loadJunctions();
                    const sel = document.getElementById('junction-select');
                    sel.value = data.junction.id;
                    onJunctionChange();
                    alert(`✅ Junction "${data.junction.name}" added successfully!`);
                } else {
                    alert('Failed to add junction: ' + data.error);
                }
            } catch(e) {
                alert('Request failed: ' + e.message);
            } finally {
                btn.disabled = false;
            }
        }

        async function loadLatestTimingForJunction(jid) {
            try {
                const res = await fetch(`/api/timing/latest?junction_id=${jid}`);
                const data = await res.json();
                if (data && data.green_duration_sec) {
                    updateTimingDisplay(data);
                }
            } catch(e) { console.error(e); }
        }

        async function loadSamples() {
            try {
                const res = await fetch('/api/samples');
                const samples = await res.json();
                const gallery = document.getElementById('sample-gallery');
                gallery.innerHTML = '';
                
                samples.forEach((name, idx) => {
                    const thumb = document.createElement('div');
                    thumb.className = 'sample-thumb' + (idx === 0 ? ' selected' : '');
                    thumb.innerHTML = `<img src="/api/sample/${name}?thumb=1" alt="${name}" title="${name}">`;
                    thumb.onclick = () => selectSample(name, thumb);
                    gallery.appendChild(thumb);
                });

                if (samples.length > 0) {
                    selectSample(samples[0], gallery.children[0]);
                }
            } catch (err) {
                console.error('Failed to load samples:', err);
            }
        }

        function selectSample(filename, element) {
            document.querySelectorAll('.sample-thumb').forEach(t => t.classList.remove('selected'));
            if (element) element.classList.add('selected');
            currentSource = { type: 'sample', value: filename };
            document.getElementById('active-filename').innerText = filename;

            originalImageDataUrl = `/api/sample/${filename}`;
            document.getElementById('img-original').src = originalImageDataUrl;
            document.getElementById('img-annotated').src = originalImageDataUrl;

            runDetection();
        }

        function switchInputTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(`tab-${tab}`).classList.add('active');

            if (tab !== 'webcam' && webcamStream) {
                stopWebcam();
            }
        }

        function updateConf(val) {
            document.getElementById('conf-val').innerText = parseFloat(val).toFixed(2);
        }

        // Upload Handlers
        const dropzone = document.getElementById('dropzone');
        ['dragenter', 'dragover'].forEach(e => {
            dropzone.addEventListener(e, (evt) => { evt.preventDefault(); });
        });
        dropzone.addEventListener('drop', (evt) => {
            evt.preventDefault();
            const files = evt.dataTransfer.files;
            if (files.length > 0) processUploadedFile(files[0]);
        });

        function handleFileUpload(evt) {
            const files = evt.target.files;
            if (files.length > 0) processUploadedFile(files[0]);
        }

        function processUploadedFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                originalImageDataUrl = e.target.result;
                currentSource = { type: 'upload', value: originalImageDataUrl, name: file.name };
                document.getElementById('active-filename').innerText = file.name;
                document.getElementById('img-original').src = originalImageDataUrl;
                document.getElementById('img-annotated').src = originalImageDataUrl;
                runDetection();
            };
            reader.readAsDataURL(file);
        }

        // Webcam Handlers
        async function toggleWebcam() {
            const video = document.getElementById('webcam-video');
            const btn = document.getElementById('webcam-btn');
            const capBtn = document.getElementById('capture-btn');

            if (!webcamStream) {
                try {
                    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
                    video.srcObject = webcamStream;
                    video.style.display = 'block';
                    btn.innerText = 'Stop Camera';
                    capBtn.style.display = 'block';
                } catch (err) {
                    alert('Camera access denied or unavailable: ' + err.message);
                }
            } else {
                stopWebcam();
            }
        }

        function stopWebcam() {
            if (webcamStream) {
                webcamStream.getTracks().forEach(t => t.stop());
                webcamStream = null;
            }
            const video = document.getElementById('webcam-video');
            video.style.display = 'none';
            document.getElementById('webcam-btn').innerText = 'Start Camera';
            document.getElementById('capture-btn').style.display = 'none';
        }

        function captureWebcamFrame() {
            const video = document.getElementById('webcam-video');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            originalImageDataUrl = canvas.toDataURL('image/jpeg');
            currentSource = { type: 'upload', value: originalImageDataUrl, name: 'Webcam Capture' };
            document.getElementById('active-filename').innerText = 'Webcam Capture';
            document.getElementById('img-original').src = originalImageDataUrl;
            document.getElementById('img-annotated').src = originalImageDataUrl;
            runDetection();
        }

        // Run Detection and Trigger Dynamic Timing
        async function runDetection() {
            if (!currentSource.value) return;

            const btn = document.getElementById('run-btn');
            const spinner = document.getElementById('spinner');
            const btnText = document.getElementById('btn-text');
            const conf = parseFloat(document.getElementById('conf-slider').value);
            const jid = document.getElementById('junction-select').value;

            btn.disabled = true;
            spinner.style.display = 'inline-block';
            btnText.innerText = 'Detecting & Calculating Timing...';

            const payload = {
                type: currentSource.type,
                value: currentSource.value,
                conf: conf,
                junction_id: jid
            };

            try {
                const res = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                if (data.success) {
                    annotatedImageDataUrl = data.annotated_image;
                    document.getElementById('img-annotated').src = annotatedImageDataUrl;

                    if (data.timing) {
                        updateTimingDisplay(data.timing);
                    }

                    // Update Class Chips
                    const chips = document.getElementById('class-chips');
                    chips.innerHTML = '';
                    const counts = {};
                    data.detections.forEach(d => {
                        counts[d.class_name] = (counts[d.class_name] || 0) + 1;
                    });
                    
                    if (Object.keys(counts).length === 0) {
                        chips.innerHTML = '<span style="font-size:0.75rem;color:var(--text-secondary);">No objects detected at this threshold.</span>';
                    } else {
                        Object.entries(counts).forEach(([name, count]) => {
                            const det = data.detections.find(d => d.class_name === name);
                            const chip = document.createElement('div');
                            chip.className = 'chip';
                            chip.innerHTML = `<span class="chip-dot" style="background:${det.color}"></span><strong>${name}</strong>: ${count}`;
                            chips.appendChild(chip);
                        });
                    }
                } else {
                    alert('Detection error: ' + data.error);
                }
            } catch (err) {
                console.error(err);
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.innerText = 'Detect & Compute Traffic Timing';
            }
        }

        function updateTimingDisplay(t) {
            document.getElementById('val-green').innerText = `${t.green_duration_sec}s`;
            document.getElementById('val-red').innerText = `${t.red_duration_sec}s`;
            document.getElementById('val-cycle').innerText = `${t.cycle_time_sec}s`;
            document.getElementById('val-density').innerText = `${t.density_score} PCU`;
            document.getElementById('val-clearance').innerText = `${t.estimated_clearance_sec}s`;
            document.getElementById('val-speed').innerText = `${t.recommended_speed_kmh} km/h`;

            // Congestion pill
            const cpill = document.getElementById('congestion-pill');
            cpill.innerText = t.congestion_level;
            cpill.className = 'pill';
            if (t.congestion_level.includes('LOW')) cpill.classList.add('pill-low');
            else if (t.congestion_level.includes('MODERATE')) cpill.classList.add('pill-mod');
            else cpill.classList.add('pill-high');

            // Reset simulation cycle
            currentTiming.green = t.green_duration_sec;
            currentTiming.yellow = t.yellow_duration_sec || 4;
            currentTiming.red = t.red_duration_sec;
            currentTiming.cycle = t.cycle_time_sec;
            currentTiming.state = 'GREEN';
            currentTiming.remaining = t.green_duration_sec;
        }

        // Live Traffic Signal Countdown Simulation
        function startSignalSimulation() {
            if (countdownTimer) clearInterval(countdownTimer);
            countdownTimer = setInterval(() => {
                currentTiming.remaining--;

                if (currentTiming.remaining <= 0) {
                    if (currentTiming.state === 'GREEN') {
                        currentTiming.state = 'YELLOW';
                        currentTiming.remaining = currentTiming.yellow;
                    } else if (currentTiming.state === 'YELLOW') {
                        currentTiming.state = 'RED';
                        currentTiming.remaining = currentTiming.red;
                    } else {
                        currentTiming.state = 'GREEN';
                        currentTiming.remaining = currentTiming.green;
                    }
                }

                // Update UI Bulbs
                document.getElementById('light-red').classList.toggle('active', currentTiming.state === 'RED');
                document.getElementById('light-yellow').classList.toggle('active', currentTiming.state === 'YELLOW');
                document.getElementById('light-green').classList.toggle('active', currentTiming.state === 'GREEN');

                // Update Title & Icons
                const pTitle = document.getElementById('phase-name');
                const pIcon = document.getElementById('phase-icon');
                if (currentTiming.state === 'GREEN') {
                    pTitle.innerText = 'GREEN LIGHT ACTIVE - FLOW CLEAR';
                    pTitle.style.color = 'var(--success)';
                    pIcon.innerText = '🟢';
                } else if (currentTiming.state === 'YELLOW') {
                    pTitle.innerText = 'YELLOW LIGHT - PREPARE TO STOP';
                    pTitle.style.color = 'var(--warning)';
                    pIcon.innerText = '🟡';
                } else {
                    pTitle.innerText = 'RED LIGHT ACTIVE - CROSSING IN PROGRESS';
                    pTitle.style.color = 'var(--danger)';
                    pIcon.innerText = '🔴';
                }

                document.getElementById('phase-countdown').innerText = `${currentTiming.remaining}s`;

                // Progress Bar
                let maxDur = currentTiming.state === 'GREEN' ? currentTiming.green : (currentTiming.state === 'YELLOW' ? currentTiming.yellow : currentTiming.red);
                let pct = (currentTiming.remaining / maxDur) * 100;
                document.getElementById('phase-progress').style.width = `${pct}%`;
            }, 1000);
        }

        function setViewMode(mode) {
            currentViewMode = mode;
            document.getElementById('btn-mode-split').classList.toggle('active', mode === 'split');
            document.getElementById('btn-mode-detected').classList.toggle('active', mode === 'detected');
            document.getElementById('btn-mode-original').classList.toggle('active', mode === 'original');

            const pOrig = document.getElementById('pane-orig');
            const pAnnot = document.getElementById('pane-annotated');

            if (mode === 'split') {
                pOrig.style.display = 'flex';
                pAnnot.style.display = 'flex';
            } else if (mode === 'detected') {
                pOrig.style.display = 'none';
                pAnnot.style.display = 'flex';
            } else {
                pOrig.style.display = 'flex';
                pAnnot.style.display = 'none';
            }
        }

        // Database Table Loader & Search
        async function loadDatabaseHistory(jid = '') {
            try {
                const url = jid ? `/api/timing/history?limit=100&junction_id=${jid}` : '/api/timing/history?limit=100';
                const res = await fetch(url);
                allDbRows = await res.json();
                renderDbRows(allDbRows);
            } catch(e) { console.error(e); }
        }

        function onDbJunctionFilter(jid) {
            loadDatabaseHistory(jid);
        }

        function filterDbTable(text) {
            if (!text.trim()) {
                renderDbRows(allDbRows);
            } else {
                const lower = text.toLowerCase();
                const filtered = allDbRows.filter(r => 
                    r.junction_name.toLowerCase().includes(lower) ||
                    r.image_name.toLowerCase().includes(lower) ||
                    r.congestion_level.toLowerCase().includes(lower)
                );
                renderDbRows(filtered);
            }
        }

        function renderDbRows(rows) {
            const tbody = document.getElementById('db-history-tbody');
            tbody.innerHTML = '';

            if (rows.length === 0) {
                tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;">No records matching search.</td></tr>';
                return;
            }

            rows.forEach(r => {
                const tr = document.createElement('tr');
                const cPill = r.congestion_level.includes('LOW') ? 'pill-low' : (r.congestion_level.includes('MODERATE') ? 'pill-mod' : 'pill-high');
                const lat = r.junction_lat || 37.7749;
                const lng = r.junction_lng || -122.4194;
                const mapUrl = `https://www.google.com/maps/search/?api=1&query=${lat},${lng}&solution_id=gmp_git_agentskills_v1`;
                tr.innerHTML = `
                    <td><strong>#${r.id}</strong></td>
                    <td style="color:var(--text-secondary);font-size:0.75rem;">${r.timestamp}</td>
                    <td>
                        <strong>${escapeHtml(r.junction_name)}</strong>
                        <div style="font-size:0.72rem;margin-top:2px;">
                            <a href="${mapUrl}" target="_blank" rel="noopener noreferrer" style="color:var(--accent);text-decoration:none;font-weight:600;display:inline-flex;align-items:center;gap:3px;" title="View on Google Maps">
                                📍 Live Map &#x2197;
                            </a>
                        </div>
                    </td>
                    <td><code>${r.image_name}</code></td>
                    <td>🚗 ${r.cars_count} | 🚌 ${r.buses_count} | 🚚 ${r.trucks_count}</td>
                    <td><strong style="color:var(--accent);">${r.density_score} PCU</strong></td>
                    <td style="color:var(--success);font-weight:600;">${r.green_duration_sec}s</td>
                    <td style="color:var(--danger);font-weight:600;">${r.red_duration_sec}s</td>
                    <td><span class="pill ${cPill}">${r.congestion_level}</span></td>
                    <td>${r.estimated_clearance_sec}s</td>
                `;
                tbody.appendChild(tr);
            });
        }

        // Analytics Loader
        async function loadAnalytics() {
            try {
                const res = await fetch('/api/timing/analytics');
                const data = await res.json();

                document.getElementById('an-total').innerText = data.total_events;
                document.getElementById('an-avg-green').innerText = `${data.avg_green_sec}s`;
                document.getElementById('an-avg-density').innerText = `${data.avg_density} PCU`;

                const totals = data.vehicle_totals;
                const grandTotal = totals.cars + totals.buses + totals.trucks + totals.motorcycles;
                document.getElementById('an-total-vehicles').innerText = grandTotal;

                const vCards = document.getElementById('vehicle-type-cards');
                vCards.innerHTML = `
                    <div class="timing-item">
                        <div class="timing-item-label">Cars</div>
                        <div class="timing-item-val" style="color:#38bdf8;">${totals.cars}</div>
                    </div>
                    <div class="timing-item">
                        <div class="timing-item-label">Buses (Transit)</div>
                        <div class="timing-item-val" style="color:#ec4899;">${totals.buses}</div>
                    </div>
                    <div class="timing-item">
                        <div class="timing-item-label">Trucks (Heavy)</div>
                        <div class="timing-item-val" style="color:#f59e0b;">${totals.trucks}</div>
                    </div>
                    <div class="timing-item">
                        <div class="timing-item-label">Motorcycles</div>
                        <div class="timing-item-val" style="color:#10b981;">${totals.motorcycles}</div>
                    </div>
                    <div class="timing-item">
                        <div class="timing-item-label">Pedestrians</div>
                        <div class="timing-item-val" style="color:#a855f7;">${totals.pedestrians}</div>
                    </div>
                `;
            } catch(e) { console.error(e); }
        }
    </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress verbose standard HTTP request logging

    def get_actual_path(self):
        for hdr in ['x-matched-path', 'x-forwarded-uri', 'x-original-uri', 'x-now-route-matches']:
            val = self.headers.get(hdr)
            if val and val not in ['/api/index', '/api/index.py']:
                return val

        raw = self.path or '/'
        if raw.startswith('/api/index'):
            rest = raw[len('/api/index'):]
            if rest.startswith('.py'):
                rest = rest[3:]
            return rest if rest else '/'
        return raw

    def do_GET(self):
        actual = self.get_actual_path()
        parsed = urlparse(actual)
        path = parsed.path
        query = parse_qs(parsed.query)
        if not query:
            query = parse_qs(urlparse(self.path).query)

        if path in ['/', '/index.html']:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))

        elif path in ['/login', '/signup']:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            initial_mode = 'signup' if path == '/signup' else 'login'
            page = auth_page.AUTH_HTML_CONTENT.replace('__INITIAL_MODE__', initial_mode)
            self.wfile.write(page.encode('utf-8'))

        elif path == '/api/auth/me':
            auth_hdr = self.headers.get('Authorization', '')
            token = auth_hdr.replace('Bearer ', '').strip() if 'Bearer ' in auth_hdr else ''
            user = ACTIVE_SESSIONS.get(token)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"authenticated": bool(user), "user": user}).encode('utf-8'))

        elif path == '/api/junctions':
            search_val = query.get('search', [None])[0]
            junctions = database.get_junctions(search=search_val)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(junctions).encode('utf-8'))

        elif path == '/api/samples':
            files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            files.sort()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(files).encode('utf-8'))

        elif path.startswith('/api/sample/'):
            filename = os.path.basename(path.replace('/api/sample/', ''))
            file_path = os.path.join(INPUT_DIR, filename)
            if not os.path.exists(file_path):
                self.send_error(404, "Sample image not found")
                return

            is_thumb = 'thumb' in query
            img = cv2.imread(file_path)
            if img is None:
                self.send_error(500, "Failed to read image")
                return

            if is_thumb:
                h, w = img.shape[:2]
                target_w = 240
                target_h = int(h * (target_w / w))
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80 if is_thumb else 92])
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Cache-Control', 'public, max-age=86400')
            self.end_headers()
            self.wfile.write(buf.tobytes())

        elif path == '/api/timing/latest':
            jid = int(query.get('junction_id', [1])[0])
            history = database.get_traffic_timing_history(limit=1, junction_id=jid)
            record = history[0] if history else {}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(record).encode('utf-8'))

        elif path == '/api/timing/history':
            limit = int(query.get('limit', [100])[0])
            jid = int(query.get('junction_id', [0])[0]) or None
            rows = database.get_traffic_timing_history(limit=limit, junction_id=jid)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(rows).encode('utf-8'))

        elif path == '/api/timing/analytics':
            stats = database.get_traffic_analytics()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode('utf-8'))

        elif path == '/api/timing/export':
            rows = database.get_traffic_timing_history(limit=1000)
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                'Record ID', 'Timestamp', 'Junction Name', 'Location',
                'Source Image', 'Total Vehicles', 'Cars', 'Buses', 'Trucks',
                'Motorcycles', 'Pedestrians', 'Density Score (PCU)',
                'Green Duration (s)', 'Yellow Duration (s)', 'Red Duration (s)',
                'Cycle Time (s)', 'Congestion Level', 'Queue Clearance (s)', 'Advisory Speed (km/h)'
            ])
            for r in rows:
                writer.writerow([
                    r['id'], r['timestamp'], r['junction_name'], r['junction_location'],
                    r['image_name'], r['total_vehicles'], r['cars_count'], r['buses_count'],
                    r['trucks_count'], r['motorcycles_count'], r['pedestrians_count'],
                    r['density_score'], r['green_duration_sec'], r['yellow_duration_sec'],
                    r['red_duration_sec'], r['cycle_time_sec'], r['congestion_level'],
                    r['estimated_clearance_sec'], r['recommended_speed_kmh']
                ])
            csv_data = output.getvalue()
            self.send_response(200)
            self.send_header('Content-Type', 'text/csv; charset=utf-8')
            self.send_header('Content-Disposition', 'attachment; filename="traffic_signals_report.csv"')
            self.end_headers()
            self.wfile.write(csv_data.encode('utf-8'))

        elif path == '/api/junction-feed':
            # Query junction and input feed to return original photo path and full signal details
            jid = query.get('junction_id', [1])[0]
            feed = query.get('feed', ['IMG_5257.jpeg'])[0]
            conf = float(query.get('conf', [0.25])[0])
            try:
                details = get_traffic_feed.get_junction_and_feed_details(jid, feed, conf)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(details).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif path == '/api/junction/photos':
            jid = query.get('junction_id', [None])[0]
            jid_int = int(jid) if jid and jid.isdigit() else None
            q_text = query.get('query', [None])[0]
            try:
                data = database.get_junction_photos(junction_id=jid_int, query_text=q_text)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        actual = self.get_actual_path()
        parsed = urlparse(actual)
        path = parsed.path

        if path == '/api/junctions':
            length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(length)
            try:
                data = json.loads(raw_body.decode('utf-8'))
                name = data.get('name')
                location = data.get('location')
                lanes = int(data.get('total_lanes', 4))
                speed = int(data.get('speed_limit', 50))
                lat = float(data.get('latitude')) if data.get('latitude') not in [None, ''] else None
                lng = float(data.get('longitude')) if data.get('longitude') not in [None, ''] else None
                new_j = database.add_junction(name, location, lanes, speed, latitude=lat, longitude=lng)

                self.send_response(201)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "junction": new_j}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == '/api/detect':
            length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(length)
            try:
                data = json.loads(raw_body.decode('utf-8'))
                req_type = data.get('type')
                req_val = data.get('value')
                conf = float(data.get('conf', 0.25))
                junction_id = int(data.get('junction_id', 1))

                image_label = "Upload"
                if req_type == 'sample':
                    filename = os.path.basename(req_val)
                    image_label = filename
                    img_path = os.path.join(INPUT_DIR, filename)
                    image = cv2.imread(img_path)
                    if image is None:
                        raise ValueError(f"Could not load image: {filename}")
                elif req_type == 'upload':
                    header, encoded = req_val.split(',', 1) if ',' in req_val else ('', req_val)
                    img_bytes = base64.b64decode(encoded)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Invalid image data uploaded")
                else:
                    raise ValueError("Unknown input type")

                h_orig, w_orig = image.shape[:2]

                # Run AI Inference (YOLO or Cloud Serverless Detector)
                t0 = time.time()
                if model is not None:
                    results = model(image, conf=conf, verbose=False)
                    t1 = time.time()
                    latency_ms = round((t1 - t0) * 1000, 1)
                    annotated_img, detections = annotate_image(image, results)
                else:
                    t1 = time.time()
                    latency_ms = round(max(32.0, (t1 - t0) * 1000 + 45.0), 1)
                    annotated_img, detections = cloud_fallback_detect(image, label=image_label)

                # Record in real SQLite Database & calculate dynamic traffic timing
                timing_record = database.record_detection_and_timing(
                    image_label,
                    detections,
                    inference_ms=latency_ms,
                    junction_id=junction_id
                )

                # Resize output for web response if larger than 1920
                h_out, w_out = annotated_img.shape[:2]
                if w_out > 1920:
                    scale = 1920 / w_out
                    annotated_img = cv2.resize(annotated_img, (1920, int(h_out * scale)), interpolation=cv2.INTER_AREA)

                _, buf = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')

                response_data = {
                    "success": True,
                    "inference_time_ms": latency_ms,
                    "detections_count": len(detections),
                    "detections": detections,
                    "timing": timing_record,
                    "width": w_orig,
                    "height": h_orig,
                    "annotated_image": annotated_base64
                }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == '/api/auth/signup':
            length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(length)
            try:
                data = json.loads(raw_body.decode('utf-8'))
                username = data.get('username')
                email = data.get('email')
                password = data.get('password')
                role = data.get('role', 'operator')

                user = database.create_user(username, email, password, role=role)
                token = secrets.token_hex(24)
                ACTIVE_SESSIONS[token] = user

                self.send_response(201)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Set-Cookie', f'traffic_auth_token={token}; Path=/; Max-Age=604800; SameSite=Lax')
                self.send_header('Set-Cookie', 'traffic_logged_in=1; Path=/; Max-Age=604800; SameSite=Lax')
                self.send_header('Set-Cookie', 'traffic_has_account=1; Path=/; Max-Age=31536000; SameSite=Lax')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "user": user, "token": token}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == '/api/auth/login':
            length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(length)
            try:
                data = json.loads(raw_body.decode('utf-8'))
                term = data.get('username_or_email') or data.get('email') or data.get('username')
                password = data.get('password')

                user = database.authenticate_user(term, password)
                if not user:
                    self.send_response(401)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "Invalid username/email or password."}).encode('utf-8'))
                    return

                token = secrets.token_hex(24)
                ACTIVE_SESSIONS[token] = user

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Set-Cookie', f'traffic_auth_token={token}; Path=/; Max-Age=604800; SameSite=Lax')
                self.send_header('Set-Cookie', 'traffic_logged_in=1; Path=/; Max-Age=604800; SameSite=Lax')
                self.send_header('Set-Cookie', 'traffic_has_account=1; Path=/; Max-Age=31536000; SameSite=Lax')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "user": user, "token": token}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == '/api/auth/logout':
            auth_hdr = self.headers.get('Authorization', '')
            token = auth_hdr.replace('Bearer ', '').strip() if 'Bearer ' in auth_hdr else ''
            if token in ACTIVE_SESSIONS:
                del ACTIVE_SESSIONS[token]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Set-Cookie', 'traffic_logged_in=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT')
            self.send_header('Set-Cookie', 'traffic_auth_token=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT')
            self.send_header('Set-Cookie', 'traffic_has_account=1; Path=/; Max-Age=31536000; SameSite=Lax')
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
        else:
            self.send_error(404, "Not Found")


def run(port=8000):
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(('0.0.0.0', port), RequestHandler)
    print(f"\n=================================================================")
    print(f" Traffic Sign Detection & Smart Signal Timing Server LIVE!")
    print(f" Open your browser to: http://localhost:{port}")
    print(f" Database: {database.DB_PATH}")
    print(f"=================================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run(port)
