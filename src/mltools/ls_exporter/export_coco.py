import os
import json


def _results_to_coco(results, include_masks=True) -> dict:
    """
    Pure transform: converts a list of Ultralytics Result objects to a COCO dict.
    No file I/O.
    """
    # ==========================================
    # Phase 2: Core Data Initialization
    # ==========================================
    coco_format = {
        "info": {"description": "YOLO to COCO Export Pipeline", "version": "1.0", "year": 2026},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    current_image_id = 1
    current_annotation_id = 1

    # Extract class dictionary from the first result object
    for class_id, class_name in results[0].names.items():
        coco_format["categories"].append({
            "id": int(class_id),
            "name": class_name,
            "supercategory": "none"
        })

    # ==========================================
    # Phase 3: The Parsing Loop
    # ==========================================
    for result in results:
        # Image Metadata
        original_height, original_width = result.orig_shape[:2]
        filename = os.path.basename(result.path)

        coco_format["images"].append({
            "id": current_image_id,
            "file_name": filename,
            "width": int(original_width),
            "height": int(original_height)
        })

        has_masks = result.masks is not None

        # Annotation Extraction
        for i, box in enumerate(result.boxes):
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            width = x_max - x_min
            height = y_max - y_min

            segmentation = []
            if include_masks and has_masks:
                polygon = result.masks.xy[i]
                flat_polygon = [round(float(coord), 2) for coord in polygon.reshape(-1)]
                if len(flat_polygon) >= 6:
                    segmentation = [flat_polygon]

            coco_format["annotations"].append({
                "id": current_annotation_id,
                "image_id": current_image_id,
                "category_id": int(box.cls[0].item()),
                "bbox": [round(x_min, 2), round(y_min, 2), round(width, 2), round(height, 2)],
                "area": round(width * height, 2),
                "iscrowd": 0,
                "segmentation": segmentation,
                "score": round(float(box.conf[0].item()), 4)
            })
            current_annotation_id += 1

        current_image_id += 1

    # Phase 4: filter unused categories
    used_ids = {ann["category_id"] for ann in coco_format["annotations"]}
    coco_format["categories"] = [c for c in coco_format["categories"] if c["id"] in used_ids]

    return coco_format


def export_annotations(results, output_json_path, validate=False, debug=False, include_masks=True, indent=4):
    """
    Exports Ultralytics YOLO inference results to a COCO-formatted JSON file.

    Args:
        results (list): A list of Ultralytics Result objects returned by model(image).
        output_json_path (str): The desired file path for the exported JSON.
        validate (bool): If True, uses pycocotools to validate the JSON structure.
        debug (bool): If True, validates the JSON and displays a matplotlib plot of the first image.
        include_masks (bool): If True, extracts and includes polygon segmentation masks if available.
        indent (int, optional): JSON indentation level. Set to None for minified output.
    """
    # Phase 1: guard
    if not results:
        print("Warning: Empty results list provided. No JSON exported.")
        return

    coco_format = _results_to_coco(results, include_masks)

    # ==========================================
    # Phase 4: File Serialization
    # ==========================================
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=indent)

    print(f"Successfully exported {len(coco_format['annotations'])} annotations to '{output_json_path}'")

    # ==========================================
    # Phase 5: The Flag Triggers
    # ==========================================
    if validate or debug:
        try:
            from pycocotools.coco import COCO
            print("\nValidating COCO JSON...")
            coco = COCO(output_json_path)
            print("Validation successful!")

            if debug:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches

                print("\nOpening debug visualization...")
                img_id = coco.getImgIds()[0]
                img_info = coco.loadImgs(img_id)[0]
                image_filename = img_info['file_name']

                if not os.path.exists(image_filename):
                    print(f"Cannot visualize: '{image_filename}' not found.")
                    return

                image = plt.imread(image_filename)
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(image)
                ax.axis('off')

                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)

                coco.showAnns(anns, draw_bbox=False)

                for ann in anns:
                    top_left_x, top_left_y, w, h = ann['bbox']
                    cat_name = coco.loadCats(ann['category_id'])[0]['name']

                    rect = patches.Rectangle((top_left_x, top_left_y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(top_left_x, top_left_y - 5, f"{cat_name.upper()}", color='black', fontsize=10, weight='bold',
                            bbox=dict(facecolor='lime', edgecolor='none', alpha=0.8, pad=2))

                plt.title(f"Debug View: {image_filename}")
                plt.tight_layout()
                plt.show()

        except ImportError:
            print("\nNotice: 'pycocotools' or 'matplotlib' is not installed. Skipping validation/debug.")
        except Exception as e:
            print(f"\nValidation Error: {e}")
