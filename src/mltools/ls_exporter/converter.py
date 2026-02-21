import uuid

def coco_to_ls_tasks(coco: dict, image_server_url: str, include_boxes: bool = False) -> list[dict]:
    img_map = {img["id"]: img for img in coco.get("images", [])}
    cat_map = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    ann_by_img: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    tasks = []
    base_url = image_server_url.rstrip("/")

    for img_id, img in img_map.items():
        w, h = img["width"], img["height"]
        file_name = img["file_name"]

        results = []
        for ann in ann_by_img.get(img_id, []):
            bx, by, bw, bh = ann["bbox"]
            
            if include_boxes:
                results.append({
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": w,
                    "original_height": h,
                    "value": {
                        "x": bx / w * 100,
                        "y": by / h * 100,
                        "width": bw / w * 100,
                        "height": bh / h * 100,
                        "rotation": 0,
                        "rectanglelabels": [cat_map[ann["category_id"]]],
                    },
                    "score": ann.get("score", 1.0),
                })

            seg = ann.get("segmentation", [])
            if seg and seg[0]:
                flat_coords = seg[0]
                points = [
                    [flat_coords[j] / w * 100, flat_coords[j + 1] / h * 100]
                    for j in range(0, len(flat_coords) - 1, 2)
                ]
                region_id = uuid.uuid4().hex[:8]
                results.append({
                    "id": region_id,
                    "type": "polygonlabels",
                    "from_name": "mask",
                    "to_name": "image",
                    "original_width": w,
                    "original_height": h,
                    "value": {
                        "points": points,
                        "polygonlabels": [cat_map[ann["category_id"]]],
                    },
                    "score": ann.get("score", 1.0),
                })
                
                results.append({
                    "id": region_id,
                    "type": "number",
                    "from_name": "score",
                    "to_name": "image",
                    "value": {"number": round(ann.get("score", 1.0), 3)},
                })

        tasks.append({
            "data": {
                "image": f"{base_url}/{file_name}",
            },
            "predictions": [
                {
                    "model_version": "yolo11n",
                    "score": max((r["score"] for r in results if "score" in r), default=0),
                    "result": results,
                }
            ] if results else [],
        })

    return tasks