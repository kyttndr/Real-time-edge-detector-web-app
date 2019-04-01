import sys
import logging

logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, "/var/www/html/Real-time-edge-detector-web-app/")

from app import app as application
