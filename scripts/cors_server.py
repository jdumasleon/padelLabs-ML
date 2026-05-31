#!/usr/bin/env python3
"""
Simple HTTP server with CORS headers — serves labeled-strokes/ CSV files to Label Studio.
Run from the PadelLabs-ML directory:
    python3 scripts/cors_server.py
"""
import http.server
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labeled-strokes")
PORT = 8090

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress per-request logs


if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", PORT), CORSHandler)
    print(f"Serving {ROOT} at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()
