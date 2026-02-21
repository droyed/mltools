from functools import partial
from http.server import SimpleHTTPRequestHandler, HTTPServer

class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):
        # Keep the test output clean by hiding standard HTTP GET logs
        pass

def start_server(port: int, directory: str):
    handler = partial(CORSHandler, directory=directory)
    server = HTTPServer(("0.0.0.0", port), handler)
    print(f"[image-server] Serving {directory} on http://localhost:{port}")
    server.serve_forever()