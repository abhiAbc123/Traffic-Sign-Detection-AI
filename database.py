import os
import sqlite3
import json
import hashlib
import secrets
from datetime import datetime
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# On Vercel / serverless runtimes, only /tmp is writable
if os.environ.get('VERCEL'):
    TMP_DB = '/tmp/traffic_data.db'
    if not os.path.exists(TMP_DB):
        orig_db = os.path.join(BASE_DIR, 'traffic_data.db')
        if os.path.exists(orig_db):
            try:
                import shutil
                shutil.copyfile(orig_db, TMP_DB)
            except Exception:
                pass
    DB_PATH = TMP_DB
else:
    DB_PATH = os.path.join(BASE_DIR, 'traffic_data.db')

INPUT_DIR = os.path.join(BASE_DIR, 'imageRecognition', 'input_images')


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table 1: Junctions / Camera Locations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS junctions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            total_lanes INTEGER DEFAULT 4,
            speed_limit INTEGER DEFAULT 50,
            latitude REAL DEFAULT 37.7749,
            longitude REAL DEFAULT -122.4194,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Migration for latitude & longitude columns
    cursor.execute("PRAGMA table_info(junctions)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if 'latitude' not in existing_cols:
        cursor.execute("ALTER TABLE junctions ADD COLUMN latitude REAL DEFAULT 37.7749")
    if 'longitude' not in existing_cols:
        cursor.execute("ALTER TABLE junctions ADD COLUMN longitude REAL DEFAULT -122.4194")

    # Table 2: Detection Events
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            junction_id INTEGER NOT NULL,
            image_name TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_vehicles INTEGER DEFAULT 0,
            cars_count INTEGER DEFAULT 0,
            buses_count INTEGER DEFAULT 0,
            trucks_count INTEGER DEFAULT 0,
            motorcycles_count INTEGER DEFAULT 0,
            pedestrians_count INTEGER DEFAULT 0,
            signs_count INTEGER DEFAULT 0,
            signs_list TEXT,
            inference_ms REAL,
            FOREIGN KEY (junction_id) REFERENCES junctions (id)
        )
    ''')

    # Table 3: Traffic Timings (Calculated Smart Signal Timings)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_timings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            junction_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            congestion_level TEXT NOT NULL,
            density_score REAL NOT NULL,
            green_duration_sec INTEGER NOT NULL,
            yellow_duration_sec INTEGER DEFAULT 4,
            red_duration_sec INTEGER NOT NULL,
            cycle_time_sec INTEGER NOT NULL,
            estimated_clearance_sec REAL NOT NULL,
            recommended_speed_kmh INTEGER,
            signal_state TEXT DEFAULT 'GREEN',
            FOREIGN KEY (event_id) REFERENCES detection_events (id),
            FOREIGN KEY (junction_id) REFERENCES junctions (id)
        )
    ''')

    # Table 4: Users (Authentication & Roles)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT DEFAULT 'operator',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Insert default junctions if empty
    cursor.execute('SELECT COUNT(*) FROM junctions')
    if cursor.fetchone()[0] == 0:
        default_junctions = [
            ("Junction A - Central Boulevard", "Downtown Core (North-South)", 4, 50, 37.7749, -122.4194),
            ("Junction B - Highway 101 Interchange", "Expressway Corridor (East-West)", 6, 80, 37.6879, -122.4702),
            ("Junction C - Metro Station Crossing", "Transit Hub & Pedestrian Zone", 3, 40, 37.7879, -122.4075),
            ("Junction D - Tech Park Avenue", "Suburban Business District", 4, 60, 37.3861, -122.0839),
        ]
        cursor.executemany(
            'INSERT INTO junctions (name, location, total_lanes, speed_limit, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?)',
            default_junctions
        )
    else:
        # Update existing default records with specific coordinates
        cursor.execute("UPDATE junctions SET latitude = 37.7749, longitude = -122.4194 WHERE name LIKE '%Central%'")
        cursor.execute("UPDATE junctions SET latitude = 37.6879, longitude = -122.4702 WHERE name LIKE '%Highway 101%'")
        cursor.execute("UPDATE junctions SET latitude = 37.7879, longitude = -122.4075 WHERE name LIKE '%Metro%'")
        cursor.execute("UPDATE junctions SET latitude = 37.3861, longitude = -122.0839 WHERE name LIKE '%Tech Park%'")

    # Insert default demo user if empty
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        demo_salt = secrets.token_hex(16)
        demo_hash = hashlib.sha256((demo_salt + "traffic2026").encode('utf-8')).hexdigest()
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, salt, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ("demo_operator", "operator@trafficsign.ai", demo_hash, demo_salt, "operator"))

    conn.commit()
    conn.close()
    print(f"[DB] Initialized database at {DB_PATH}")


def calculate_traffic_timing(counts):
    """
    Smart Traffic Signal Timing Algorithm:
    Allocates dynamic Green / Red split based on weighted vehicle density.
    Cars: 1.0 weight
    Motorcycles: 0.5 weight
    Buses: 2.5 weight (heavy vehicle / public transit priority)
    Trucks: 3.0 weight (slow acceleration / clearance requirement)
    """
    cars = counts.get('car', 0)
    motos = counts.get('motorcycle', 0)
    buses = counts.get('bus', 0)
    trucks = counts.get('truck', 0)
    peds = counts.get('person', 0)

    # Calculate weighted traffic density score
    density_score = (cars * 1.0) + (motos * 0.5) + (buses * 2.5) + (trucks * 3.0)

    # Congestion Level categorisation
    if density_score < 4.0:
        congestion_level = "LOW (Smooth)"
        base_green = 18
    elif density_score < 12.0:
        congestion_level = "MODERATE (Steady)"
        base_green = 32
    elif density_score < 24.0:
        congestion_level = "HIGH (Busy)"
        base_green = 48
    else:
        congestion_level = "CRITICAL (Congested)"
        base_green = 65

    # Dynamically scale green duration (clamped between 15s and 90s)
    green_duration = int(min(90, max(15, base_green + (density_score * 1.8))))
    yellow_duration = 4
    # Red duration complements the opposite phase cycle
    cycle_time = 90  # 90-second standard base cycle
    red_duration = max(15, cycle_time - green_duration - yellow_duration)
    total_cycle = green_duration + yellow_duration + red_duration

    # Queue clearance time: ~2.1 seconds per equivalent passenger car unit
    clearance_sec = round(density_score * 2.1, 1)

    # Recommended approach speed
    if density_score > 20:
        rec_speed = 30
    elif density_score > 10:
        rec_speed = 40
    else:
        rec_speed = 50

    return {
        "density_score": round(density_score, 2),
        "congestion_level": congestion_level,
        "green_duration_sec": green_duration,
        "yellow_duration_sec": yellow_duration,
        "red_duration_sec": red_duration,
        "cycle_time_sec": total_cycle,
        "estimated_clearance_sec": clearance_sec,
        "recommended_speed_kmh": rec_speed,
        "signal_state": "GREEN"
    }


def record_detection_and_timing(image_name, detections, inference_ms=0.0, junction_id=1):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Count categories
    counts = {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0, 'person': 0, 'signs': 0}
    signs_found = []

    traffic_sign_names = {
        'traffic light', 'stop sign', 'Give way', 'No entry', 'One-way traffic',
        'Speed limit', 'Turn left', 'Turn right', 'No parking', 'Pedestrian crossing'
    }

    for d in detections:
        c_name = d.get('class_name', '').lower()
        if 'car' in c_name:
            counts['car'] += 1
        elif 'bus' in c_name:
            counts['bus'] += 1
        elif 'truck' in c_name:
            counts['truck'] += 1
        elif 'motorcycle' in c_name or 'bicycle' in c_name:
            counts['motorcycle'] += 1
        elif 'person' in c_name:
            counts['person'] += 1
        else:
            counts['signs'] += 1
            signs_found.append(d.get('class_name'))

    total_vehicles = counts['car'] + counts['bus'] + counts['truck'] + counts['motorcycle']

    # 1. Insert Detection Event
    cursor.execute('''
        INSERT INTO detection_events (
            junction_id, image_name, total_vehicles, cars_count,
            buses_count, trucks_count, motorcycles_count, pedestrians_count,
            signs_count, signs_list, inference_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        junction_id, image_name, total_vehicles, counts['car'],
        counts['bus'], counts['truck'], counts['motorcycle'], counts['person'],
        counts['signs'], json.dumps(signs_found), inference_ms
    ))
    event_id = cursor.lastrowid

    # 2. Calculate and Insert Traffic Timing
    timing = calculate_traffic_timing(counts)
    cursor.execute('''
        INSERT INTO traffic_timings (
            event_id, junction_id, congestion_level, density_score,
            green_duration_sec, yellow_duration_sec, red_duration_sec,
            cycle_time_sec, estimated_clearance_sec, recommended_speed_kmh,
            signal_state
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        event_id, junction_id, timing['congestion_level'], timing['density_score'],
        timing['green_duration_sec'], timing['yellow_duration_sec'], timing['red_duration_sec'],
        timing['cycle_time_sec'], timing['estimated_clearance_sec'], timing['recommended_speed_kmh'],
        timing['signal_state']
    ))
    timing_id = cursor.lastrowid

    conn.commit()

    # Fetch created record
    cursor.execute('''
        SELECT t.*, e.image_name, e.total_vehicles, e.cars_count, e.buses_count,
               e.trucks_count, e.motorcycles_count, e.pedestrians_count,
               j.name as junction_name, j.location as junction_location,
               j.latitude as junction_lat, j.longitude as junction_lng
        FROM traffic_timings t
        JOIN detection_events e ON t.event_id = e.id
        JOIN junctions j ON t.junction_id = j.id
        WHERE t.id = ?
    ''', (timing_id,))
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else timing


def get_junctions(search=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if search and search.strip():
        pattern = f"%{search.strip()}%"
        cursor.execute(
            'SELECT * FROM junctions WHERE name LIKE ? OR location LIKE ? ORDER BY id ASC',
            (pattern, pattern)
        )
    else:
        cursor.execute('SELECT * FROM junctions ORDER BY id ASC')
    raw_rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    rows = []
    for r in raw_rows:
        lat = r.get('latitude') if r.get('latitude') is not None else 37.7749
        lng = r.get('longitude') if r.get('longitude') is not None else -122.4194
        r['latitude'] = lat
        r['longitude'] = lng
        r['maps_url'] = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}&solution_id=gmp_git_agentskills_v1"
        r['embed_url'] = f"https://maps.google.com/maps?q={lat},{lng}&hl=en&z=16&output=embed"
        r['directions_url'] = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}&travelmode=driving&solution_id=gmp_git_agentskills_v1"
        rows.append(r)
    return rows


def add_junction(name, location, total_lanes=4, speed_limit=50, latitude=None, longitude=None):
    if not name or not name.strip():
        raise ValueError("Junction name cannot be empty")
    if not location or not location.strip():
        raise ValueError("Location description cannot be empty")

    lat_val = float(latitude) if latitude is not None and str(latitude).strip() != '' else 37.7749
    lng_val = float(longitude) if longitude is not None and str(longitude).strip() != '' else -122.4194

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO junctions (name, location, total_lanes, speed_limit, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name.strip(), location.strip(), int(total_lanes or 4), int(speed_limit or 50), lat_val, lng_val))
    junction_id = cursor.lastrowid
    conn.commit()

    # Seed with an initial baseline timing so signal starts active immediately
    default_timing = calculate_traffic_timing({'car': 2})
    cursor.execute('''
        INSERT INTO detection_events (
            junction_id, image_name, total_vehicles, cars_count,
            buses_count, trucks_count, motorcycles_count, pedestrians_count,
            signs_count, signs_list, inference_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (junction_id, "initial_setup", 2, 2, 0, 0, 0, 0, 0, "[]", 0.0))
    event_id = cursor.lastrowid

    cursor.execute('''
        INSERT INTO traffic_timings (
            event_id, junction_id, congestion_level, density_score,
            green_duration_sec, yellow_duration_sec, red_duration_sec,
            cycle_time_sec, estimated_clearance_sec, recommended_speed_kmh,
            signal_state
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        event_id, junction_id, default_timing['congestion_level'], default_timing['density_score'],
        default_timing['green_duration_sec'], default_timing['yellow_duration_sec'], default_timing['red_duration_sec'],
        default_timing['cycle_time_sec'], default_timing['estimated_clearance_sec'], default_timing['recommended_speed_kmh'],
        default_timing['signal_state']
    ))
    conn.commit()

    cursor.execute('SELECT * FROM junctions WHERE id = ?', (junction_id,))
    new_junction = dict(cursor.fetchone())
    conn.close()
    lat = new_junction.get('latitude') or 37.7749
    lng = new_junction.get('longitude') or -122.4194
    new_junction['latitude'] = lat
    new_junction['longitude'] = lng
    new_junction['maps_url'] = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}&solution_id=gmp_git_agentskills_v1"
    new_junction['embed_url'] = f"https://maps.google.com/maps?q={lat},{lng}&hl=en&z=16&output=embed"
    new_junction['directions_url'] = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}&travelmode=driving&solution_id=gmp_git_agentskills_v1"
    return new_junction


def get_junction_photos(junction_id=None, query_text=None):
    conn = get_db_connection()
    cursor = conn.cursor()

    junction = None
    if junction_id:
        cursor.execute('SELECT * FROM junctions WHERE id = ?', (junction_id,))
        row = cursor.fetchone()
        if row:
            junction = dict(row)

    if not junction and query_text:
        term = f"%{query_text.strip()}%"
        cursor.execute('SELECT * FROM junctions WHERE name LIKE ? OR location LIKE ? LIMIT 1', (term, term))
        row = cursor.fetchone()
        if row:
            junction = dict(row)

    # If it's a custom queried location not in local DB
    if not junction and query_text:
        clean_q = query_text.strip()
        encoded_q = quote(clean_q)
        conn.close()
        all_samples = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(INPUT_DIR) else []
        return {
            "name": clean_q,
            "location": "Custom Searched Location",
            "is_custom": True,
            "latitude": None,
            "longitude": None,
            "maps_url": f"https://www.google.com/maps/search/?api=1&query={encoded_q}&solution_id=gmp_git_agentskills_v1",
            "street_view_url": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={encoded_q}&solution_id=gmp_git_agentskills_v1",
            "photos_url": f"https://www.google.com/maps/search/?api=1&query={encoded_q}&solution_id=gmp_git_agentskills_v1",
            "directions_url": f"https://www.google.com/maps/dir/?api=1&destination={encoded_q}&travelmode=driving&solution_id=gmp_git_agentskills_v1",
            "embed_views": {
                "street_view": f"https://maps.google.com/maps?q={encoded_q}&layer=c&output=svembed",
                "satellite": f"https://maps.google.com/maps?q={encoded_q}&t=k&z=19&output=embed",
                "hybrid": f"https://maps.google.com/maps?q={encoded_q}&t=h&z=18&output=embed",
                "roadmap": f"https://maps.google.com/maps?q={encoded_q}&hl=en&z=16&output=embed"
            },
            "camera_photos": all_samples[:4]
        }

    if not junction:
        cursor.execute('SELECT * FROM junctions ORDER BY id ASC LIMIT 1')
        row = cursor.fetchone()
        junction = dict(row) if row else {
            "id": 1, "name": "Default Junction", "location": "City Center", "latitude": 37.7749, "longitude": -122.4194
        }

    jid = junction['id']
    cursor.execute('''
        SELECT DISTINCT image_name FROM detection_events
        WHERE junction_id = ? AND image_name IS NOT NULL AND image_name != '' AND image_name != 'unit_test_img.jpg' AND image_name != 'test.jpg'
        ORDER BY id DESC LIMIT 6
    ''', (jid,))
    events = cursor.fetchall()
    conn.close()

    camera_photos = [e['image_name'] for e in events]
    all_samples = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(INPUT_DIR) else []
    if not camera_photos and all_samples:
        offset = ((jid - 1) * 3) % max(1, len(all_samples))
        camera_photos = all_samples[offset:offset + 4]

    lat = junction.get('latitude') or 37.7749
    lng = junction.get('longitude') or -122.4194
    loc_query = f"{lat},{lng}"

    return {
        "junction": junction,
        "name": junction['name'],
        "location": junction['location'],
        "is_custom": False,
        "latitude": lat,
        "longitude": lng,
        "maps_url": f"https://www.google.com/maps/search/?api=1&query={loc_query}&solution_id=gmp_git_agentskills_v1",
        "street_view_url": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={loc_query}&solution_id=gmp_git_agentskills_v1",
        "photos_url": f"https://www.google.com/maps/search/?api=1&query={loc_query}&solution_id=gmp_git_agentskills_v1",
        "directions_url": f"https://www.google.com/maps/dir/?api=1&destination={loc_query}&travelmode=driving&solution_id=gmp_git_agentskills_v1",
        "embed_views": {
            "street_view": f"https://maps.google.com/maps?q=&layer=c&cbll={lat},{lng}&cbp=12,0,0,0,0&output=svembed",
            "satellite": f"https://maps.google.com/maps?q={lat},{lng}&t=k&z=19&output=embed",
            "hybrid": f"https://maps.google.com/maps?q={lat},{lng}&t=h&z=18&output=embed",
            "roadmap": f"https://maps.google.com/maps?q={lat},{lng}&hl=en&z=16&output=embed"
        },
        "camera_photos": camera_photos
    }



def get_traffic_timing_history(limit=50, junction_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = '''
        SELECT t.*, e.image_name, e.total_vehicles, e.cars_count, e.buses_count,
               e.trucks_count, e.motorcycles_count, e.pedestrians_count,
               j.name as junction_name, j.location as junction_location,
               j.latitude as junction_lat, j.longitude as junction_lng
        FROM traffic_timings t
        JOIN detection_events e ON t.event_id = e.id
        JOIN junctions j ON t.junction_id = j.id
    '''
    params = []
    if junction_id:
        query += ' WHERE t.junction_id = ?'
        params.append(junction_id)

    query += ' ORDER BY t.timestamp DESC, t.id DESC LIMIT ?'
    params.append(limit)

    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_traffic_analytics():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) as total_events FROM detection_events')
    total_events = cursor.fetchone()['total_events']

    cursor.execute('SELECT AVG(green_duration_sec) as avg_green, AVG(density_score) as avg_density FROM traffic_timings')
    avg_row = cursor.fetchone()
    avg_green = round(avg_row['avg_green'] or 0, 1)
    avg_density = round(avg_row['avg_density'] or 0, 1)

    cursor.execute('''
        SELECT SUM(cars_count) as cars, SUM(buses_count) as buses,
               SUM(trucks_count) as trucks, SUM(motorcycles_count) as motos,
               SUM(pedestrians_count) as peds
        FROM detection_events
    ''')
    sums = cursor.fetchone()

    cursor.execute('''
        SELECT congestion_level, COUNT(*) as count
        FROM traffic_timings
        GROUP BY congestion_level
    ''')
    congestion_breakdown = {r['congestion_level']: r['count'] for r in cursor.fetchall()}

    conn.close()
    return {
        "total_events": total_events,
        "avg_green_sec": avg_green,
        "avg_density": avg_density,
        "vehicle_totals": {
            "cars": sums['cars'] or 0,
            "buses": sums['buses'] or 0,
            "trucks": sums['trucks'] or 0,
            "motorcycles": sums['motos'] or 0,
            "pedestrians": sums['peds'] or 0
        },
        "congestion_breakdown": congestion_breakdown
    }


# ==========================================
# User Authentication & Management
# ==========================================

def hash_password(password, salt=None):
    if not salt:
        salt = secrets.token_hex(16)
    hash_val = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    return hash_val, salt


def verify_password(password, stored_hash, salt):
    hash_val, _ = hash_password(password, salt)
    return secrets.compare_digest(hash_val, stored_hash)


def create_user(username, email, password, role='operator'):
    username = (username or '').strip()
    email = (email or '').strip().lower()
    password = (password or '').strip()

    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if not email or '@' not in email:
        raise ValueError("Please provide a valid email address.")
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    if cursor.fetchone():
        conn.close()
        raise ValueError("Username already exists. Please choose another.")

    cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
    if cursor.fetchone():
        conn.close()
        raise ValueError("Email already registered. Please sign in or use another.")

    pwd_hash, salt = hash_password(password)
    cursor.execute('''
        INSERT INTO users (username, email, password_hash, salt, role)
        VALUES (?, ?, ?, ?, ?)
    ''', (username, email, pwd_hash, salt, role))

    user_id = cursor.lastrowid
    conn.commit()

    cursor.execute('SELECT id, username, email, role, created_at FROM users WHERE id = ?', (user_id,))
    user = dict(cursor.fetchone())
    conn.close()
    return user


def authenticate_user(username_or_email, password):
    term = (username_or_email or '').strip()
    password = (password or '').strip()

    if not term or not password:
        return None

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users
        WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)
    ''', (term, term))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    if verify_password(password, row['password_hash'], row['salt']):
        return {
            "id": row['id'],
            "username": row['username'],
            "email": row['email'],
            "role": row['role'],
            "created_at": row['created_at']
        }
    return None


def get_user_by_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, role, created_at FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None



if __name__ == '__main__':
    init_db()
    # Test a dummy detection
    sample_res = record_detection_and_timing(
        "test.jpg",
        [
            {"class_name": "car"}, {"class_name": "car"},
            {"class_name": "bus"}, {"class_name": "traffic light"}
        ],
        inference_ms=120.5
    )
    print("[DB] Test insertion successful:")
    print(json.dumps(sample_res, indent=2))

