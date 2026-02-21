import json
import os
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from urllib.error import URLError
from datetime import datetime

from .server import start_server
from .converter import coco_to_ls_tasks
from .api import build_label_config, create_project, api_post_json

def is_port_in_use(port: int) -> bool:
    """Checks if a background service is already listening on the given port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_label_studio(ls_base: str, timeout: int = 60):
    """Polls Label Studio until it is ready to accept API requests."""
    print(f"[Runner] Waiting for Label Studio to boot at {ls_base} (this may take a few seconds)...")
    start_time = time.time()
    url = f"{ls_base.rstrip('/')}/api/version"
    
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    print("[Runner] Label Studio is online and ready!")
                    return
        except (URLError, ConnectionError):
            time.sleep(1)
            
    raise TimeoutError(f"Label Studio did not start within {timeout} seconds.")

def run(name: str, 
        json_path: str, 
        image_dir: str, 
        port: int = 8888, 
        ls_base: str = "http://localhost:8081", 
        token: str = None, 
        include_boxes: bool = False) -> None:
    
    # 1. Parameter Validation
    if not token:
        token = os.getenv("LABELSTUDIO_TOKEN")
        if not token:
            raise ValueError("Label Studio token must be provided or set via LABELSTUDIO_TOKEN env var.")

    image_server_url = f"http://localhost:{port}"

    # 2. Start Image Server (Daemon Thread)
    if is_port_in_use(port):
        print(f"[Runner] Port {port} is active. Reusing existing image server.")
    else:
        server_thread = threading.Thread(
            target=start_server, 
            args=(port, image_dir), 
            daemon=True
        )
        server_thread.start()
        time.sleep(0.5)

    # 3. Start Label Studio (Subprocess)
    ls_process = None
    ls_port = urllib.parse.urlparse(ls_base).port or 8081
    
    if is_port_in_use(ls_port):
        print(f"[Runner] Port {ls_port} is active. Assuming Label Studio is already running.")
    else:
        print(f"[Runner] Starting Label Studio on port {ls_port}...")
        
        # Pass the token into the environment so Label Studio uses it for authentication
        ls_env = os.environ.copy()
        ls_env["LABEL_STUDIO_USER_TOKEN"] = token 
        
        ls_process = subprocess.Popen(
            ["label-studio", "start", "--port", str(ls_port)],
            env=ls_env,
            stdout=subprocess.DEVNULL,  # Keep the terminal output clean
            stderr=subprocess.DEVNULL
        )
        wait_for_label_studio(ls_base)

    # 4. Parse COCO data
    timestamp = datetime.now().strftime("%Y-%b-%d_%I-%M%p")
    titled = f"{name} {timestamp}"

    with open(json_path) as f:
        coco = json.load(f)
        
    used_ids = {ann["category_id"] for ann in coco.get("annotations", [])}
    active_cats = [c for c in coco.get("categories", []) if c["id"] in used_ids]
    label_config = build_label_config(active_cats)

    # 5. Create Project
    project = create_project(title=titled, label_config=label_config, ls_base=ls_base, token=token)
    project_id = project.id
    print(f"Created project '{titled}' (ID {project_id})")

    # 6. Convert and Upload Tasks
    tasks = coco_to_ls_tasks(coco, image_server_url=image_server_url, include_boxes=include_boxes)
    print(f"Uploading {len(tasks)} tasks…")
    result = api_post_json(f"/api/projects/{project_id}/import", tasks, ls_base=ls_base, token=token)
    
    # 7. Final User Notification & Blocking Loop
    url = f"{ls_base.rstrip('/')}/projects/{project_id}/data"
    print("\n" + "="*60)
    print("🚀 Label Studio export complete!")
    print(f"🔗 View annotations here: {url}")
    print("🖼️  Servers are running in the background.")
    print("⏹️  Press Ctrl+C to stop the servers and continue testing.")
    print("="*60 + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        if ls_process:
            print("[Runner] Terminating Label Studio process...")
            ls_process.terminate()
            ls_process.wait()
        print("[Runner] Moving to next test...")