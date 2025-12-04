
import os
import json
from pathlib import Path

def generate_index():
    assets_dir = Path("docs/assets")
    if not assets_dir.exists():
        print("docs/assets directory not found.")
        return

    # Find all subdirectories that contain frames_manifest.json
    scenes = []
    for d in assets_dir.iterdir():
        if d.is_dir() and (d / "frames_manifest.json").exists():
            scenes.append(d.name)
    
    scenes.sort()

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PanoSynthVR Scene Launcher</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f5f5f7;
            color: #1d1d1f;
            margin: 0;
            padding: 40px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 18px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        h1 {
            text-align: center;
            font-weight: 700;
            margin-bottom: 40px;
        }
        .scene-list {
            display: grid;
            gap: 20px;
        }
        .scene-card {
            border: 1px solid #e5e5e5;
            border-radius: 12px;
            padding: 20px;
            transition: all 0.2s ease;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .scene-card:hover {
            border-color: #0071e3;
            box-shadow: 0 4px 12px rgba(0,113,227,0.1);
            transform: translateY(-2px);
        }
        .scene-info h3 {
            margin: 0 0 5px 0;
            font-size: 18px;
        }
        .scene-info p {
            margin: 0;
            color: #86868b;
            font-size: 14px;
        }
        .launch-btn {
            background-color: #0071e3;
            color: white;
            text-decoration: none;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: 500;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        .launch-btn:hover {
            background-color: #0077ed;
        }
        .empty-state {
            text-align: center;
            color: #86868b;
            padding: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PanoSynthVR Scenes</h1>
        <div class="scene-list">
"""

    if not scenes:
        html_content += """            <div class="empty-state">
                No scenes found in docs/assets. Run the pipeline to generate some!
            </div>"""
    else:
        for scene in scenes:
            url = f"atlas_sequence_viewer.html?scene={scene}&manifest=frames_manifest.json&foreground_manifest=foreground_manifest.json&bg_static=true"
            html_content += f"""            <div class="scene-card">
                <div class="scene-info">
                    <h3>{scene}</h3>
                    <p>Dynamic Scene</p>
                </div>
                <a href="{url}" class="launch-btn" target="_blank">Launch Viewer</a>
            </div>
"""

    html_content += """        </div>
    </div>
</body>
</html>"""

    output_path = Path("docs/index.html")
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Generated index.html with {len(scenes)} scenes at {output_path}")

if __name__ == "__main__":
    generate_index()
