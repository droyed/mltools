# mltools

Toolkit for providing extensive ML toolkit. We are starting off with tools for exporting YOLO inference results to COCO format and importing them into Label Studio for annotation review.

> **Work in progress.** APIs and module structure may change.

## Overview

`mltools` bridges Ultralytics YOLO object detection with Label Studio annotation workflows. Given a set of images and a YOLO model, it runs inference, converts the results to COCO JSON, and imports them into a Label Studio project — spinning up the image server and Label Studio instance automatically if needed.

```
YOLO inference → COCO JSON → Label Studio tasks → running LS project
```

## Features

- Export Ultralytics YOLO results (boxes + segmentation masks) to standard COCO JSON
- Convert COCO JSON to Label Studio task format with pre-annotations
- Create Label Studio projects and upload tasks via the Label Studio API
- Serve local images over HTTP with CORS support so Label Studio can load them
- Orchestrate the full pipeline in a single `run()` call

## Project Structure

```
mltools/
├── src/mltools/
│   └── ls_exporter/
│       ├── export_coco.py   # YOLO results → COCO JSON
│       ├── converter.py     # COCO JSON → Label Studio task format
│       ├── api.py           # Label Studio project/task API wrapper
│       ├── server.py        # CORS-enabled local HTTP image server
│       └── runner.py        # Full pipeline orchestration
├── demos/
│   └── test_ls_exporter.py  # End-to-end usage example
├── tests/
├── assets/
└── pyproject.toml
```

## Installation

Requires Python >= 3.9.

```bash
pip install -e .
```

For development (includes `ultralytics` and `pytest`):

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from ultralytics import YOLO
import mltools.ls_exporter as ls_exporter
from mltools.ls_exporter import run

model = YOLO("yolo11n-seg.pt")
results = model(image_paths, imgsz=1280, conf=0.05)

ls_exporter.export_annotations(results, "output.json")

run(
    name="my-project",
    json_path="output.json",
    image_dir=image_dir,
    token="<LS_TOKEN>",   # or set LABELSTUDIO_TOKEN env var
)
```

`run()` will start a local image server and Label Studio (if not already running), create a project, upload all tasks, and print the project URL. Press `Ctrl+C` to stop the servers.

## Modules

| Module | Purpose |
|---|---|
| `ls_exporter.export_coco` | Export Ultralytics YOLO results to COCO JSON |
| `ls_exporter.converter` | Convert COCO JSON to Label Studio task format |
| `ls_exporter.api` | Create LS projects and upload tasks via API |
| `ls_exporter.server` | CORS-enabled HTTP image server for Label Studio |
| `ls_exporter.runner` | Orchestrate the full pipeline end-to-end |

### Key signatures

```python
# export_coco.py
export_annotations(results, output_json_path, validate=False, debug=False,
                   include_masks=True, indent=4)

# runner.py
run(name, json_path, image_dir, port=8888,
    ls_base="http://localhost:8081", token=None, include_boxes=False)
```

## Dependencies

**Required** (installed automatically):
- `label-studio-sdk`

**Optional / development:**
- `ultralytics` — for running YOLO inference
- `pytest` — for running tests
- `pycocotools` — for COCO JSON validation (`validate=True`)
- `matplotlib` — for debug visualization (`debug=True`)

## Development

Run tests:

```bash
pytest
```

Run the end-to-end demo (requires `LABELSTUDIO_TOKEN` env var and test assets):

```bash
python demos/test_ls_exporter.py
```

## Attributions

Test images located at `assets/` are from the [COCO val2017 dataset](https://cocodataset.org/) (Lin et al., 2015), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
