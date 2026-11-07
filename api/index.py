import sys
import os
import traceback

# Add root directory to python path for local imports (app, database, auth_page)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from app import RequestHandler
except Exception as e:
    from http.server import BaseHTTPRequestHandler

    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"App Import Error: {e}\n{traceback.format_exc()}".encode('utf-8'))

        def do_POST(self):
            self.do_GET()


class handler(RequestHandler):
    """
    Vercel Serverless Function entrypoint.
    Inherits all routes, authentication, database endpoints,
    Google Maps live location/photos, and dynamic traffic timing.
    """
    def do_GET(self):
        try:
            super().do_GET()
        except Exception as e:
            traceback.print_exc()
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f"Serverless Runtime Error: {e}\n{traceback.format_exc()}".encode('utf-8'))
            except Exception:
                pass

    def do_POST(self):
        try:
            super().do_POST()
        except Exception as e:
            traceback.print_exc()
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f"Serverless Runtime Error: {e}\n{traceback.format_exc()}".encode('utf-8'))
            except Exception:
                pass
