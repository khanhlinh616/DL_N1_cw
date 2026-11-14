"""Two-way mapping between COCO category_id (1-90) and internal class_idx (0-79).

COCO has 80 object categories but IDs range from 1 to 90.
10 missing IDs in COCO 2017:
[12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# Official 80 COCO category IDs and names in ascending order of category_id
COCO_CLASSES_RAW: List[Tuple[int, str]] = [
    (1, "person"),
    (2, "bicycle"),
    (3, "car"),
    (4, "motorcycle"),
    (5, "airplane"),
    (6, "bus"),
    (7, "train"),
    (8, "truck"),
    (9, "boat"),
    (10, "traffic light"),
    (11, "fire hydrant"),
    (13, "stop sign"),
    (14, "parking meter"),
    (15, "bench"),
    (16, "bird"),
    (17, "cat"),
    (18, "dog"),
    (19, "horse"),
    (20, "sheep"),
    (21, "cow"),
    (22, "elephant"),
    (23, "bear"),
    (24, "zebra"),
    (25, "giraffe"),
    (27, "backpack"),
    (28, "umbrella"),
    (31, "handbag"),
    (32, "tie"),
    (33, "suitcase"),
    (34, "frisbee"),
    (35, "skis"),
    (36, "snowboard"),
    (37, "sports ball"),
    (38, "kite"),
    (39, "baseball bat"),
    (40, "baseball glove"),
    (41, "skateboard"),
    (42, "surfboard"),
    (43, "tennis racket"),
    (44, "bottle"),
    (46, "wine glass"),
    (47, "cup"),
    (48, "fork"),
    (49, "knife"),
    (50, "spoon"),
    (51, "bowl"),
    (52, "banana"),
    (53, "apple"),
    (54, "sandwich"),
    (55, "orange"),
    (56, "broccoli"),
    (57, "carrot"),
    (58, "hot dog"),
    (59, "pizza"),
    (60, "donut"),
    (61, "cake"),
    (62, "chair"),
    (63, "couch"),
    (64, "potted plant"),
    (65, "bed"),
    (67, "dining table"),
    (70, "toilet"),
    (72, "tv"),
    (73, "laptop"),
    (74, "mouse"),
    (75, "remote"),
    (76, "keyboard"),
    (77, "cell phone"),
    (78, "microwave"),
    (79, "oven"),
    (80, "toaster"),
    (81, "sink"),
    (82, "refrigerator"),
    (84, "book"),
    (85, "clock"),
    (86, "vase"),
    (87, "scissors"),
    (88, "teddy bear"),
    (89, "hair drier"),
    (90, "toothbrush"),
]

MISSING_CATEGORY_IDS: List[int] = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]


class CategoryMapper:
    """Provides bidirectional mapping between COCO category_id and class_idx."""

    def __init__(self, classes_data: Optional[List[Tuple[int, str]]] = None) -> None:
        source_data = classes_data if classes_data is not None else COCO_CLASSES_RAW
        self._cat_id_to_class_idx: Dict[int, int] = {}
        self._class_idx_to_cat_id: Dict[int, int] = {}
        self._class_idx_to_name: Dict[int, str] = {}
        self._cat_id_to_name: Dict[int, str] = {}

        for idx, (cat_id, name) in enumerate(source_data):
            self._cat_id_to_class_idx[cat_id] = idx
            self._class_idx_to_cat_id[idx] = cat_id
            self._class_idx_to_name[idx] = name
            self._cat_id_to_name[cat_id] = name

        assert len(self._cat_id_to_class_idx) == 80, f"Expected 80 classes, got {len(self._cat_id_to_class_idx)}"
        assert len(self._class_idx_to_cat_id) == 80

    def to_class_idx(self, category_id: int) -> int:
        """Map official COCO category_id (1-90) to internal class_idx (0-79)."""
        if category_id not in self._cat_id_to_class_idx:
            raise KeyError(
                f"Invalid category_id {category_id}. Must be one of the 80 valid COCO category IDs."
            )
        return self._cat_id_to_class_idx[category_id]

    def to_category_id(self, class_idx: int) -> int:
        """Map internal class_idx (0-79) back to official COCO category_id (1-90)."""
        if class_idx not in self._class_idx_to_cat_id:
            raise KeyError(f"Invalid class_idx {class_idx}. Must be within [0, 79].")
        return self._class_idx_to_cat_id[class_idx]

    def get_name(self, class_idx: int) -> str:
        """Get class name by internal class_idx."""
        return self._class_idx_to_name[class_idx]

    @property
    def class_names(self) -> List[str]:
        """List of 80 class names indexed by class_idx (0-79)."""
        return [self._class_idx_to_name[i] for i in range(80)]

    @property
    def num_classes(self) -> int:
        return 80


# Module-level default instance
DEFAULT_MAPPER = CategoryMapper()


def category_id_to_class_idx(category_id: int) -> int:
    """Convenience function for category_id -> class_idx."""
    return DEFAULT_MAPPER.to_class_idx(category_id)


def class_idx_to_category_id(class_idx: int) -> int:
    """Convenience function for class_idx -> category_id."""
    return DEFAULT_MAPPER.to_category_id(class_idx)
