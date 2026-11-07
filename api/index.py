import sys
import os

# Add root directory to python path for local imports (app, database, auth_page)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import RequestHandler

class handler(RequestHandler):
    """
    Vercel Serverless Function entrypoint.
    Inherits all routes, authentication, database endpoints,
    Google Maps live location/photos, and dynamic traffic timing.
    """
    pass

