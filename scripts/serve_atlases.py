import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import os
import shutil
import sys
import time
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def serve_and_view(output_name, frame_count, fps):
    ip = "127.0.0.1"
    port = 3600
    
    url = f"http://{ip}:{port}/docs/renderer.html?mode=sequence&name={output_name}&frames={frame_count}&fps={fps}"
    print(f"Opening viewer at: {url}")

    def start_server():
        # Serve from current directory (project root)
        server_address = (ip, port)
        httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
        print(f"Serving on {ip}:{port}...")
        httpd.serve_forever()

    threading.Thread(target=start_server, daemon=True).start()
    webbrowser.open_new(url)

    print("Press Ctrl+C to stop server")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Serve existing MCI atlases in WebXR")
    parser.add_argument("--atlas_dir", required=True, help="Directory containing atlas images (frame_*.png)")
    parser.add_argument("--name", default="served_sequence", help="Name for the served asset")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    
    args = parser.parse_args()

    input_dir = Path(args.atlas_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist.")
        return

    # Prepare destination
    dest_base = Path("docs/assets") / args.name
    dest_atlas = dest_base / "atlas"
    
    if dest_atlas.exists():
        print(f"Cleaning existing destination {dest_atlas}...")
        shutil.rmtree(dest_atlas)
    
    ensure_dir(dest_atlas)

    print(f"Copying atlases from {input_dir} to {dest_atlas}...")
    files = sorted(list(input_dir.glob("frame_*.png")))
    if not files:
        print("No frame_*.png files found!")
        return

    for f in files:
        shutil.copy(f, dest_atlas / f.name)

    print(f"Copied {len(files)} frames.")
    
    serve_and_view(args.name, len(files), args.fps)

if __name__ == "__main__":
    main()
