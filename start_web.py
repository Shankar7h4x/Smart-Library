"""Local static preview server for web assets."""

import http.server
from pathlib import Path
import socketserver

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
HOST = "127.0.0.1"
PORT = 8080


def start_server():
    WEB_ROOT.mkdir(parents=True, exist_ok=True)
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
        *args, directory=str(WEB_ROOT), **kwargs
    )
    with socketserver.TCPServer((HOST, PORT), handler) as httpd:
        print(f"Serving at http://{HOST}:{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    start_server()
