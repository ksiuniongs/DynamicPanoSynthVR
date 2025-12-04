import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json, pathlib, shutil
frames_dir = pathlib.Path("test1024")
frames = sorted(p.name for p in frames_dir.iterdir()
                if p.is_dir() and (p/"atlas.png").exists())
manifest = {"basePath": "", "frames": [f"{name}/atlas.png" for name in frames]}
out = frames_dir/"frames_manifest.json"
out.write_text(json.dumps(manifest, indent=2))
print(f"写入 {out}，共 {len(frames)} 帧")

dest = pathlib.Path("docs/assets/campus_run_1024/frames_manifest.json")
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(out, dest)
print(f"已复制到 {dest}")