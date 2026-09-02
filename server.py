#!/usr/bin/env python3
"""
Local web server for Ear Mandarin Pronunciation Tutor.
Provides proper MIME types and CORS/Cross-Origin headers for WebAssembly.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
        '.onnx': 'application/octet-stream',
        '.json': 'application/json',
        '.js': 'application/javascript',
        '.css': 'text/css',
        '.html': 'text/html',
    }

    def end_headers(self):
        # Enable SharedArrayBuffer and Cross-Origin isolation if needed by ONNX Runtime multithreading
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()


def run_server(port: int = 8000, directory: str = None):
    if directory is None:
        directory = str(Path(__file__).resolve().parent)
    os.chdir(directory)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), CustomHandler) as httpd:
        print(f"\n=======================================================")
        print(f" Ear: Mandarin Pronunciation Tutor is running locally!")
        print(f" Local Web UI: http://localhost:{port}")
        print(f" Press Ctrl+C to stop the server.")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Ear local web server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(port=args.port)
