import os
import sys

import mltools

from mltools.ls_exporter import run
from ultralytics import YOLO


def run_exporter_test():
    # Setup Paths
    assets_dir = 'assets'
    image_dir = os.path.join(assets_dir, 'images_YOLO')
    imgpath = os.path.join(image_dir, 'person_207.png')
    output_json = 'tmp_output_test_new_exporter.json'
    
    token = os.getenv("LABELSTUDIO_TOKEN")
    if not token:
        print("Error: LABELSTUDIO_TOKEN environment variable is missing.")
        sys.exit(1)

    print("Loading YOLO model...")
    model = YOLO("yolo26n-seg.pt") 

    if not os.path.exists(imgpath):
        print(f"Warning: Image not found at {imgpath}. Make sure test assets are in place.")

    print(f"Predicting on {imgpath}...")
    results = model(imgpath, imgsz=1280, conf=0.05)

    print(f"Exporting annotations to {output_json}...")
    mltools.ls_exporter.export_annotations(results, output_json)

    print("Triggering Label Studio Exporter...")
    
    # Run the exporter with injected configurations (blocks on Ctrl+C)
    run(
        name="Test Run 1", 
        json_path=output_json,
        image_dir=image_dir,
        port=8888,
        ls_base="http://localhost:8081",
        token=token,
        include_boxes=False
    )

if __name__ == "__main__":
    run_exporter_test()