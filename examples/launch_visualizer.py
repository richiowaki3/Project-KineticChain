# -*- coding: utf-8 -*-
"""
Launches the 3D Dance Kinematics & Onomatopoeia Visualizer in the user's default browser.
Can serve via local HTTP server or open standalone HTML directly.
"""

import sys
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    standalone_html = project_root / "visualizer" / "standalone_MRka5p5qTxw.html"
    vis_data = project_root / "visualizer" / "data" / "MRka5p5qTxw_vis.json"

    # If data does not exist, export it first
    if not vis_data.exists():
        print("Exporting visualizer dataset for MRka5p5qTxw...")
        from src.tracking.smpl_adapter import SmplTrackAdapter
        from src.core.pipeline import DanceKinematicsPipeline
        from src.visualization.exporter import VisualizerDataExporter

        pkl_path = Path(r"D:\motion_capture\output_results\MRka5p5qTxw__4dhumans_tracks.pkl")
        loaded = SmplTrackAdapter.load_4dhumans_pkl(pkl_path, target_track_id=1)
        pipeline = DanceKinematicsPipeline(fps=30.0, with_onomatopoeia=True)
        result = pipeline.process(loaded["joints"], loaded["poses"], video_id="MRka5p5qTxw")
        VisualizerDataExporter.export_to_json(result, vis_data)

    if not standalone_html.exists():
        print("Generating standalone HTML viewer...")
        from tools.build_standalone_viewer import build_standalone_viewer
        build_standalone_viewer(vis_data, project_root / "visualizer" / "index.html", standalone_html)

    # Option 1: Start local HTTP server to allow video streaming and clean URLs
    PORT = 8085
    vis_dir = str(project_root / "visualizer")

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=vis_dir, **kwargs)
        def log_message(self, format, *args):
            pass # Suppress noisy log outputs

    try:
        httpd = socketserver.TCPServer(("", PORT), QuietHandler)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        url = f"http://localhost:{PORT}/standalone_MRka5p5qTxw.html"
        print("=" * 70)
        print("3D Dance Kinematics & Onomatopoeia Visualizer 起動中...")
        print(f"ブラウザで以下のURLを開きます: {url}")
        print("終了するには Ctrl+C を押してください。")
        print("=" * 70)
        webbrowser.open(url)
    except Exception as e:
        # Fallback to direct file URI
        file_url = standalone_html.as_uri()
        print(f"ローカルサーバー起動スキップ ({e})。直接ブラウザでファイルを開きます:\n{file_url}")
        webbrowser.open(file_url)

    try:
        # Keep process alive for web server
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nVisualizer サーバーを停止しました。")


if __name__ == "__main__":
    main()
