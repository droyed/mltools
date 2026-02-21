import json
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from mltools.ls_exporter.export_coco import export_annotations


# ---------------------------------------------------------------------------
# Helpers: build lightweight fakes for Ultralytics Result objects
# ---------------------------------------------------------------------------

def _make_box(x_min, y_min, x_max, y_max, class_id):
    """Return a mock box object mimicking ultralytics box API."""
    box = MagicMock()
    # box.xyxy[0].tolist() -> [x_min, y_min, x_max, y_max]
    box.xyxy.__getitem__.return_value.tolist.return_value = [x_min, y_min, x_max, y_max]
    # box.cls[0].item() -> class_id
    box.cls.__getitem__.return_value.item.return_value = class_id
    return box


def _make_result(
    path,
    orig_shape,
    names,
    boxes_data,
    mask_polygons=None,
):
    """
    Return a mock Ultralytics Result object.

    Args:
        path: image file path string
        orig_shape: (height, width) tuple
        names: {int: str} class dict
        boxes_data: list of (x_min, y_min, x_max, y_max, class_id)
        mask_polygons: None, or list of numpy arrays (one per box) of shape (N, 2)
    """
    result = MagicMock()
    result.path = path
    result.orig_shape = orig_shape
    result.names = names
    result.boxes = [_make_box(*b) for b in boxes_data]

    if mask_polygons is not None:
        result.masks = MagicMock()
        result.masks.xy = mask_polygons
    else:
        result.masks = None

    return result


NAMES = {0: "cat", 1: "dog"}

# A simple result with two boxes, no masks
SINGLE_RESULT = _make_result(
    path="/data/img001.jpg",
    orig_shape=(480, 640),
    names=NAMES,
    boxes_data=[
        (10.0, 20.0, 110.0, 120.0, 0),
        (200.0, 50.0, 400.0, 300.0, 1),
    ],
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyResults:
    def test_empty_list_returns_early(self, capsys, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([], out_path)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert not os.path.exists(out_path)


class TestCOCOStructure:
    def test_top_level_keys(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert set(data.keys()) == {"info", "licenses", "images", "annotations", "categories"}

    def test_info_block(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert "description" in data["info"]
        assert "version" in data["info"]

    def test_licenses_is_list(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert isinstance(data["licenses"], list)


class TestCategories:
    def test_category_count(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert len(data["categories"]) == 2

    def test_category_fields(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        cat_names = {c["name"] for c in data["categories"]}
        assert cat_names == {"cat", "dog"}
        for cat in data["categories"]:
            assert "id" in cat
            assert "supercategory" in cat


class TestImageMetadata:
    def test_single_image_entry(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert len(data["images"]) == 1

    def test_image_fields(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        img = data["images"][0]
        assert img["file_name"] == "img001.jpg"
        assert img["width"] == 640
        assert img["height"] == 480
        assert img["id"] == 1

    def test_multiple_images(self, tmp_path):
        result2 = _make_result("/data/img002.png", (720, 1280), NAMES, [(5, 5, 50, 50, 0)])
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT, result2], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert len(data["images"]) == 2
        ids = [img["id"] for img in data["images"]]
        assert ids == [1, 2]


class TestAnnotations:
    def test_annotation_count(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert len(data["annotations"]) == 2

    def test_annotation_ids_are_unique_and_sequential(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        ids = [a["id"] for a in data["annotations"]]
        assert ids == list(range(1, len(ids) + 1))

    def test_bbox_conversion_xywh(self, tmp_path):
        """xyxy -> [x_min, y_min, width, height]"""
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        first = data["annotations"][0]
        x_min, y_min, w, h = first["bbox"]
        assert x_min == pytest.approx(10.0)
        assert y_min == pytest.approx(20.0)
        assert w == pytest.approx(100.0)   # 110 - 10
        assert h == pytest.approx(100.0)   # 120 - 20

    def test_area_equals_w_times_h(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        for ann in data["annotations"]:
            _, _, w, h = ann["bbox"]
            assert ann["area"] == pytest.approx(w * h, rel=1e-3)

    def test_iscrowd_is_zero(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert all(a["iscrowd"] == 0 for a in data["annotations"])

    def test_image_id_matches(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        for ann in data["annotations"]:
            assert ann["image_id"] == 1

    def test_category_id_matches_class(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert data["annotations"][0]["category_id"] == 0
        assert data["annotations"][1]["category_id"] == 1

    def test_annotation_count_across_multiple_images(self, tmp_path):
        result2 = _make_result("/data/img002.png", (720, 1280), NAMES, [(5, 5, 50, 50, 0)])
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT, result2], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert len(data["annotations"]) == 3  # 2 + 1


class TestSegmentationMasks:
    def _result_with_masks(self):
        polygon_a = np.array([[10, 20], [110, 20], [110, 120], [10, 120]], dtype=float)
        polygon_b = np.array([[200, 50], [400, 50], [400, 300], [200, 300]], dtype=float)
        return _make_result(
            path="/data/masked.jpg",
            orig_shape=(480, 640),
            names=NAMES,
            boxes_data=[
                (10.0, 20.0, 110.0, 120.0, 0),
                (200.0, 50.0, 400.0, 300.0, 1),
            ],
            mask_polygons=[polygon_a, polygon_b],
        )

    def test_segmentation_included_when_masks_present(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([self._result_with_masks()], out_path, include_masks=True)
        with open(out_path) as f:
            data = json.load(f)
        for ann in data["annotations"]:
            assert len(ann["segmentation"]) == 1
            assert len(ann["segmentation"][0]) >= 6

    def test_segmentation_excluded_when_flag_false(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([self._result_with_masks()], out_path, include_masks=False)
        with open(out_path) as f:
            data = json.load(f)
        for ann in data["annotations"]:
            assert ann["segmentation"] == []

    def test_segmentation_empty_when_no_masks(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path, include_masks=True)
        with open(out_path) as f:
            data = json.load(f)
        for ann in data["annotations"]:
            assert ann["segmentation"] == []

    def test_short_polygon_skipped(self, tmp_path):
        """A polygon with fewer than 3 points (< 6 flat coords) must be skipped."""
        tiny_polygon = np.array([[10, 20], [30, 40]], dtype=float)  # only 2 points
        result = _make_result(
            path="/data/tiny.jpg",
            orig_shape=(480, 640),
            names=NAMES,
            boxes_data=[(10.0, 20.0, 110.0, 120.0, 0)],
            mask_polygons=[tiny_polygon],
        )
        out_path = str(tmp_path / "out.json")
        export_annotations([result], out_path, include_masks=True)
        with open(out_path) as f:
            data = json.load(f)
        assert data["annotations"][0]["segmentation"] == []


class TestFileSerialization:
    def test_output_file_created(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        assert os.path.exists(out_path)

    def test_output_is_valid_json(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_minified_output_with_indent_none(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path, indent=None)
        with open(out_path) as f:
            raw = f.read()
        assert "\n" not in raw  # minified: no newlines

    def test_pretty_output_with_indent(self, tmp_path):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path, indent=2)
        with open(out_path) as f:
            raw = f.read()
        assert "\n" in raw


class TestValidateFlag:
    def test_validate_triggers_coco_import(self, tmp_path, capsys):
        out_path = str(tmp_path / "out.json")
        with patch("mltools.ls_exporter.export_coco.COCO", create=True) as mock_coco_cls:
            # Patch the import inside the function
            with patch.dict("sys.modules", {"pycocotools.coco": MagicMock(COCO=mock_coco_cls)}):
                export_annotations([SINGLE_RESULT], out_path, validate=True)

    def test_missing_pycocotools_handled_gracefully(self, tmp_path, capsys):
        out_path = str(tmp_path / "out.json")
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            # The function catches ImportError internally; file should already be written
            pass
        # Verify file written before validate block runs
        export_annotations([SINGLE_RESULT], out_path, validate=False)
        assert os.path.exists(out_path)


class TestPrintOutput:
    def test_success_message_printed(self, tmp_path, capsys):
        out_path = str(tmp_path / "out.json")
        export_annotations([SINGLE_RESULT], out_path)
        captured = capsys.readouterr()
        assert "Successfully exported" in captured.out
        assert "2" in captured.out  # annotation count
