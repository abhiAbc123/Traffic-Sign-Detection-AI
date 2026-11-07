# Traffic Sign & Smart Traffic Timing AI System

An end-to-end computer vision and intelligent transportation system (ITS) for traffic sign, signal, and vehicle detection using **YOLOv8** (Ultralytics & PyTorch) paired with a **real SQLite database model** (`traffic_data.db`) that computes and logs dynamic traffic signal timings.

---

## 🔗 Project Resources & Links

| Resource | Link | Description |
| :--- | :--- | :--- |
| 🚀 **Production Web App** | [https://traffic-sign-detection-ai.vercel.app/](https://traffic-sign-detection-ai.vercel.app/) | Live single-page application hosted on **Vercel** with full client-side routing, YOLO AI detection, Google Maps live locations & photos, and user auth. |
| 📁 **GitHub Source Code** | [https://github.com/abhiAbc123/Traffic-Sign-Detection-AI](https://github.com/abhiAbc123/Traffic-Sign-Detection-AI) | Public open-source repository containing frontend client, YOLOv8 computer vision engine, and SQLite database models. |

---

## 🌟 Key Features

### 1. Computer Vision & Detection Engine
- **YOLOv8 Medium Architecture**: Native detection of traffic signs, signals, cars, buses, trucks, motorcycles, and pedestrians.
- **Dynamic Input Feed**: Works on pre-loaded dataset sample images, user drag-and-drop uploads, and live webcam capture.
- **Customizable Sensitivity**: Real-time confidence threshold slider (0.05 to 0.95).

### 2. Real Database Model (`traffic_data.db`)
- Powered by a relational **SQLite** database (`database.py`) with complete relational integrity:
  - **`junctions`**: Monitored intersection corridors (lanes, speed limits, location coordinates).
  - **`detection_events`**: Detailed detection logs per camera frame (total vehicles, cars, buses, trucks, motorcycles, pedestrians, signs, inference latency).
  - **`traffic_timings`**: Dynamically computed signal timing records (green light duration, red phase, yellow clearance, congestion level, queue clearance time, advisory speed).

### 3. Dynamic Smart Traffic Signal Timing Algorithm
Traffic signal green/red phases are dynamically calculated from the detected vehicle density:
- **Weighted Density Formula**:
  $$\text{Density Score} = 1.0 \times \text{Cars} + 0.5 \times \text{Motorcycles} + 2.5 \times \text{Buses} + 3.0 \times \text{Trucks}$$
- **Adaptive Green Phase**: Scales from 15s up to 90s based on queue density.
- **Complementary Red Phase**: Balanced against standard 90s total cycle time.
- **Queue Clearance Time**: Estimates seconds required to clear the junction bottleneck.

### 4. Interactive Browser UI & User Authentication
Open **[http://localhost:8000](http://localhost:8000)** to experience:
- **Realistic 3D Traffic Light Widget**: Live glowing 🔴 Red, 🟡 Yellow, and 🟢 Green phases with real-time countdown timer.
- **Exact Location Photos & Google Maps Live Viewer**:
  - 📸 **Street View 360° Photo**: Ground-level photographic panorama of the exact junction or any searched place worldwide.
  - 🛰️ **Satellite Aerial Photos**: High-resolution overhead photography of the intersection, lanes, and roadway geometry.
  - 🚦 **CCTV Camera Feed Photos**: Real traffic camera captures recorded at the junction, with 1-click instant AI vehicle & signal detection.
  - 🗺️ **Interactive Vector Map**: Real-time road network with traffic.
  - **Live Search**: Type any junction or global location (e.g., *Times Square*, *Highway 101*, *Central Blvd*) and instantly inspect exact photos and coordinates.
- **GPS Coordinates on Custom Junctions**: Adding junctions supports manual coordinates or 1-click device GPS capture via HTML5 Geolocation.
- **Dedicated User Signup & Login (`/signup`, `/login`)**: Beautiful card design matching the `TrafficVision PRO` UI with email/username/password inputs, quick toggling, and one-click demo access (`demo_operator`).
- **Database History Table**: Inspect all historical detection events with direct Google Maps pins and computed green/red allocations.
- **Real-Time Analytics & CSV Export**: Average green duration, congestion distributions, and vehicle volume breakdown.

---

## 🚀 Quick Start

### 1. Installation & Environment Setup

```powershell
# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Launch the Web Interface

```powershell
.venv\Scripts\python.exe app.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

### 3. Command-Line Detection

To run batch inference across images in the terminal:
```powershell
.venv\Scripts\python.exe imageRecognition/main.py --conf 0.25
```

---

## 📡 REST API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main Web Application UI |
| `/login` | `GET` | Dedicated User Login Page |
| `/signup` | `GET` | Dedicated User Signup Page |
| `/api/auth/signup` | `POST` | Register a new user (`email`, `username`, `password`, `role`) |
| `/api/auth/login` | `POST` | Authenticate user credentials (`username_or_email`, `password`) |
| `/api/auth/me` | `GET` | Query current user profile via Bearer token |
| `/api/auth/logout` | `POST` | Invalidate active user session |
| `/api/junctions` | `GET` | List of all monitored traffic junctions (supports `?search=<query>`) |
| `/api/junctions` | `POST` | Create a new monitoring junction (`name`, `location`, `total_lanes`, `speed_limit`) |
| `/api/junction/photos` | `GET` | Retrieve exact location photos (Street View 360°, Satellite, CCTV camera feeds) for any junction or searched place (`?junction_id=<id>&query=<location>`) |
| `/api/samples` | `GET` | List of test dataset images |
| `/api/sample/<filename>` | `GET` | Fetch raw image or thumbnail (`?thumb=1`) |
| `/api/detect` | `POST` | Run YOLO detection, insert database event, and compute signal timings |
| `/api/timing/latest` | `GET` | Get the latest timing record for a junction (`?junction_id=1`) |
| `/api/timing/history` | `GET` | Query past database records (`?limit=50&junction_id=1`) |
| `/api/timing/analytics` | `GET` | Aggregate statistics, total vehicles, and congestion breakdown |
| `/api/timing/export` | `GET` | Export complete database history as CSV report |

---

## 📁 Repository Structure

```
Traffic-Sign-Detection-AI/
├── app.py                         # Web server + REST API + Frontend UI
├── auth_page.py                   # Dedicated User Signup & Login template
├── database.py                    # SQLite database models (junctions, detections, timings, users)
├── get_traffic_feed.py            # Standalone junction feed query utility
├── seed_db.py                     # Database population script
├── traffic_data.db                # SQLite database storing real traffic timing history & user accounts
├── requirements.txt               # Pinned dependencies
├── tests/
│   └── test_traffic_system.py     # 14 unit tests (database, timing formulas, auth & APIs)
├── README.md                      # Documentation
└── imageRecognition/
    ├── main.py                    # Batch CLI detection runner
    ├── yolov8m.pt                 # YOLOv8 medium model weights
    ├── data.yaml                  # Dataset configuration
    ├── input_images/              # 20 raw test images
    └── DetectionOutput/           # Annotated detection output images
```
