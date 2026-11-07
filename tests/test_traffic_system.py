import os
import unittest
import urllib.request
import json
import database


class TestTrafficTimingDatabase(unittest.TestCase):
    def setUp(self):
        database.init_db()

    def test_junctions_exist(self):
        junctions = database.get_junctions()
        self.assertGreaterEqual(len(junctions), 4)
        names = [j['name'] for j in junctions]
        self.assertTrue(any('Central Boulevard' in n for n in names))

    def test_timing_calculation_zero_traffic(self):
        # Empty road
        timing = database.calculate_traffic_timing({})
        self.assertEqual(timing['density_score'], 0.0)
        self.assertIn('LOW', timing['congestion_level'])
        self.assertGreaterEqual(timing['green_duration_sec'], 15)
        self.assertLessEqual(timing['green_duration_sec'], 25)

    def test_timing_calculation_moderate_traffic(self):
        # 5 cars, 1 bus
        counts = {'car': 5, 'bus': 1}
        # score = 5*1.0 + 1*2.5 = 7.5
        timing = database.calculate_traffic_timing(counts)
        self.assertEqual(timing['density_score'], 7.5)
        self.assertIn('MODERATE', timing['congestion_level'])
        self.assertGreater(timing['green_duration_sec'], 30)

    def test_timing_calculation_heavy_traffic(self):
        # 15 cars, 3 buses, 2 trucks
        counts = {'car': 15, 'bus': 3, 'truck': 2}
        # score = 15 + 7.5 + 6.0 = 28.5
        timing = database.calculate_traffic_timing(counts)
        self.assertGreaterEqual(timing['density_score'], 24.0)
        self.assertIn('CRITICAL', timing['congestion_level'])
        self.assertGreaterEqual(timing['green_duration_sec'], 65)

    def test_record_and_query_detection(self):
        sample_dets = [
            {"class_name": "car", "confidence": 0.85},
            {"class_name": "truck", "confidence": 0.78},
            {"class_name": "traffic light", "confidence": 0.92}
        ]
        rec = database.record_detection_and_timing(
            image_name="unit_test_img.jpg",
            detections=sample_dets,
            inference_ms=88.4,
            junction_id=1
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec['image_name'], "unit_test_img.jpg")
        self.assertEqual(rec['total_vehicles'], 2)
        self.assertEqual(rec['cars_count'], 1)
        self.assertEqual(rec['trucks_count'], 1)

        # Check history retrieval
        history = database.get_traffic_timing_history(limit=5, junction_id=1)
        self.assertGreater(len(history), 0)
        latest = history[0]
        self.assertEqual(latest['image_name'], "unit_test_img.jpg")

    def test_analytics_metrics(self):
        analytics = database.get_traffic_analytics()
        self.assertIn('total_events', analytics)
        self.assertGreater(analytics['total_events'], 0)
        self.assertIn('vehicle_totals', analytics)
        self.assertIn('cars', analytics['vehicle_totals'])

    def test_database_user_auth(self):
        # Verify default demo operator
        demo = database.authenticate_user("demo_operator", "traffic2026")
        self.assertIsNotNone(demo)
        self.assertEqual(demo['username'], "demo_operator")
        self.assertEqual(demo['role'], "operator")

        # Verify authentication by email
        demo_email = database.authenticate_user("operator@trafficsign.ai", "traffic2026")
        self.assertIsNotNone(demo_email)
        self.assertEqual(demo_email['id'], demo['id'])

        # Verify wrong password fails
        bad_auth = database.authenticate_user("demo_operator", "wrong_password")
        self.assertIsNone(bad_auth)

        # Test creating a unique test user
        import secrets
        uid = secrets.token_hex(4)
        new_user = database.create_user(f"officer_{uid}", f"officer_{uid}@traffic.ai", "securePass2026")
        self.assertIsNotNone(new_user)
        self.assertEqual(new_user['username'], f"officer_{uid}")

        # Test duplicate creation raises ValueError
        with self.assertRaises(ValueError):
            database.create_user(f"officer_{uid}", f"another_{uid}@traffic.ai", "securePass2026")


class TestWebAPIEndpoints(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8000"

    def fetch_json(self, path):
        req = urllib.request.Request(f"{self.BASE_URL}{path}")
        with urllib.request.urlopen(req, timeout=5) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read().decode('utf-8'))

    def test_add_and_search_junctions(self):
        new_j = database.add_junction("Test Plaza Crossing", "Uptown Sector 7", 4, 45, latitude=34.0522, longitude=-118.2437)
        self.assertIsNotNone(new_j['id'])
        self.assertEqual(new_j['name'], "Test Plaza Crossing")
        self.assertAlmostEqual(new_j['latitude'], 34.0522, places=4)
        self.assertAlmostEqual(new_j['longitude'], -118.2437, places=4)
        self.assertIn("maps.google.com", new_j['embed_url'])
        self.assertIn("gmp_git_agentskills_v1", new_j['maps_url'])

        results = database.get_junctions(search="Plaza")
        self.assertTrue(any(j['name'] == "Test Plaza Crossing" for j in results))

    def test_api_add_and_search_junctions(self):
        payload = json.dumps({
            "name": "API Test Boulevard",
            "location": "Sector 9 South",
            "total_lanes": 6,
            "speed_limit": 60,
            "latitude": 40.7128,
            "longitude": -74.0060
        }).encode('utf-8')
        req = urllib.request.Request(
            f"{self.BASE_URL}/api/junctions",
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            self.assertEqual(response.status, 201)
            data = json.loads(response.read().decode('utf-8'))
            self.assertTrue(data['success'])
            self.assertEqual(data['junction']['name'], "API Test Boulevard")
            self.assertIn("google.com/maps", data['junction']['maps_url'])

        # Test search
        search_res = self.fetch_json("/api/junctions?search=Boulevard")
        self.assertTrue(any(j['name'] == "API Test Boulevard" for j in search_res))


    def test_api_samples(self):
        data = self.fetch_json("/api/samples")
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_api_timing_latest(self):
        data = self.fetch_json("/api/timing/latest?junction_id=1")
        self.assertIn('green_duration_sec', data)
        self.assertIn('congestion_level', data)

    def test_api_timing_history(self):
        data = self.fetch_json("/api/timing/history?limit=10")
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_api_timing_analytics(self):
        data = self.fetch_json("/api/timing/analytics")
        self.assertIn('total_events', data)
        self.assertIn('avg_green_sec', data)

    def test_api_auth_pages_and_endpoints(self):
        # 0. GET / (Dashboard Auth Gate check)
        req_root = urllib.request.Request(f"{self.BASE_URL}/")
        with urllib.request.urlopen(req_root, timeout=5) as res:
            self.assertEqual(res.status, 200)
            root_html = res.read().decode('utf-8')
            self.assertIn("traffic_user", root_html)
            self.assertIn("/signup", root_html)
            self.assertIn("/login", root_html)
            self.assertIn("traffic_has_account", root_html)

        # 1. GET /login
        req_login = urllib.request.Request(f"{self.BASE_URL}/login")
        with urllib.request.urlopen(req_login, timeout=5) as res:
            self.assertEqual(res.status, 200)
            html = res.read().decode('utf-8')
            self.assertIn("TrafficVision", html)
            self.assertIn("PRO", html)
            self.assertIn("User Login", html)
            self.assertIn("traffic_user", html)

        # 2. GET /signup
        req_signup = urllib.request.Request(f"{self.BASE_URL}/signup")
        with urllib.request.urlopen(req_signup, timeout=5) as res:
            self.assertEqual(res.status, 200)
            html = res.read().decode('utf-8')
            self.assertIn("TrafficVision", html)
            self.assertIn("User Signup", html)
            self.assertIn("traffic_user", html)

        # 3. POST /api/auth/login (Demo Operator)
        login_payload = json.dumps({
            "username_or_email": "demo_operator",
            "password": "traffic2026"
        }).encode('utf-8')
        req_auth = urllib.request.Request(
            f"{self.BASE_URL}/api/auth/login",
            data=login_payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req_auth, timeout=5) as res:
            self.assertEqual(res.status, 200)
            cookie_headers = res.headers.get_all('Set-Cookie') or []
            cookie_str = " ".join(cookie_headers)
            self.assertIn("traffic_logged_in=1", cookie_str)
            self.assertIn("traffic_has_account=1", cookie_str)
            data = json.loads(res.read().decode('utf-8'))
            self.assertTrue(data['success'])
            self.assertEqual(data['user']['username'], "demo_operator")
            self.assertTrue('token' in data)
            token = data['token']

        # 4. GET /api/auth/me with Bearer token
        req_me = urllib.request.Request(
            f"{self.BASE_URL}/api/auth/me",
            headers={'Authorization': f'Bearer {token}'}
        )
        with urllib.request.urlopen(req_me, timeout=5) as res:
            self.assertEqual(res.status, 200)
            me_data = json.loads(res.read().decode('utf-8'))
            self.assertTrue(me_data['authenticated'])
            self.assertEqual(me_data['user']['username'], "demo_operator")

        # 5. POST /api/auth/signup
        import secrets
        uid = secrets.token_hex(4)
        signup_payload = json.dumps({
            "email": f"tester_{uid}@trafficsign.ai",
            "username": f"tester_{uid}",
            "password": "strongPassword123"
        }).encode('utf-8')
        req_new_user = urllib.request.Request(
            f"{self.BASE_URL}/api/auth/signup",
            data=signup_payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req_new_user, timeout=5) as res:
            self.assertEqual(res.status, 201)
            cookie_headers = res.headers.get_all('Set-Cookie') or []
            cookie_str = " ".join(cookie_headers)
            self.assertIn("traffic_logged_in=1", cookie_str)
            self.assertIn("traffic_has_account=1", cookie_str)
            new_data = json.loads(res.read().decode('utf-8'))
            self.assertTrue(new_data['success'])
            self.assertEqual(new_data['user']['username'], f"tester_{uid}")

        # 6. POST /api/auth/logout
        req_logout = urllib.request.Request(
            f"{self.BASE_URL}/api/auth/logout",
            headers={'Authorization': f'Bearer {token}'},
            method='POST'
        )
        with urllib.request.urlopen(req_logout, timeout=5) as res:
            self.assertEqual(res.status, 200)
            cookie_headers = res.headers.get_all('Set-Cookie') or []
            cookie_str = " ".join(cookie_headers)
            self.assertIn("traffic_has_account=1", cookie_str)
            logout_data = json.loads(res.read().decode('utf-8'))
            self.assertTrue(logout_data['success'])

    def test_api_junction_photos(self):
        # 1. Query photos for junction 1
        data = self.fetch_json("/api/junction/photos?junction_id=1")
        self.assertIn("embed_views", data)
        self.assertIn("street_view", data["embed_views"])
        self.assertIn("satellite", data["embed_views"])
        self.assertIn("camera_photos", data)
        self.assertGreater(len(data["camera_photos"]), 0)

        # 2. Query custom place search photos
        custom_data = self.fetch_json("/api/junction/photos?query=Times%20Square")
        self.assertEqual(custom_data["name"], "Times Square")
        self.assertTrue(custom_data["is_custom"])
        self.assertIn("Times%20Square", custom_data["embed_views"]["street_view"])
        self.assertIn("Times%20Square", custom_data["embed_views"]["satellite"])


if __name__ == '__main__':
    unittest.main()

