"""Shared constants for the walk_guide training pipeline."""

# YOLO class order. The Flutter app maps detections by these names, so keep them unchanged.
CLASS_NAMES = [
    "braille_block",        # 0  점자블록 (조각)
    "person",               # 1  사람
    "obstacle",             # 2  장애물 (볼라드 제외 28종 통합)
    "bollard",              # 3  볼라드 - 따로 학습해서 인식률을 올린다 (앱에서는 장애물로 취급)
    "road",                 # 4  차도·자전거도로·골목길 (조각)
    "crosswalk",            # 5  횡단보도 (조각)
    "caution_zone",         # 6  발밑 위험 구역: 계단·맨홀·배수구 덮개·가로수 보호판·공사 구역 (조각)
    "braille_damaged",      # 7  훼손된 점자블록 (조각)
    "traffic_light_red",    # 8  빨간불 신호등
    "traffic_light_green",  # 9  초록불 신호등
]
(BRAILLE, PERSON, OBSTACLE, BOLLARD, ROAD, CROSSWALK, CAUTION, BRAILLE_DAMAGED,
 TL_RED, TL_GREEN) = range(len(CLASS_NAMES))

# Classes that come from 서피스마스킹 polygons (cut into slices) and from 바운딩박스 boxes.
SURFACE_CLASSES = {BRAILLE, ROAD, CROSSWALK, CAUTION, BRAILLE_DAMAGED}
BOX_CLASSES = {PERSON, OBSTACLE, BOLLARD, TL_RED, TL_GREEN}

# AI Hub 인도보행 영상 (dataSetSn=189) bounding-box labels.
PERSON_LABELS = {"person"}
OBSTACLE_LABELS = {
    # moving objects (person is handled separately)
    "bicycle", "bus", "car", "carrier", "cat", "dog", "motorcycle", "movable_signage",
    "scooter", "stroller", "truck", "wheelchair",
    # fixed objects (bollard and traffic_light are handled separately)
    "barricade", "bench", "chair", "fire_hydrant", "kiosk", "parking_meter", "pole",
    "potted_plant", "power_controller", "stop", "table",
    "traffic_light_controller", "traffic_sign", "tree_trunk",
}
TRAFFIC_LIGHT = "traffic_light"  # colour is decided from the pixels in convert_aihub.py

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# File name prefixes in the converted dataset, used later by pseudo_label.py.
SOURCE_BBOX = "B"     # came from the 바운딩박스 set: box classes labeled, surface classes missing
SOURCE_SURFACE = "S"  # came from the 서피스마스킹 set: surface classes labeled, box classes missing


def _norm(text: str) -> str:
    return (text or "").strip().lower().replace(" ", "_").replace("-", "_")


def map_box_label(label: str):
    """YOLO class id for a 바운딩박스 <box> label, TRAFFIC_LIGHT for traffic lights, or None."""
    name = _norm(label)
    if name in PERSON_LABELS:
        return PERSON
    if name == "bollard":
        return BOLLARD
    if name == TRAFFIC_LIGHT:
        return TRAFFIC_LIGHT
    if name in OBSTACLE_LABELS:
        return OBSTACLE
    return None


def map_surface_label(label: str, attribute: str = ""):
    """YOLO class id for a 서피스마스킹 <polygon> (label + its attribute), or None to ignore it."""
    name, attr = _norm(label), _norm(attribute)
    if "braille" in name:
        return BRAILLE_DAMAGED if attr == "damaged" else BRAILLE
    if name in ("roadway", "alley"):
        return CROSSWALK if attr == "crosswalk" else ROAD
    if name == "bike_lane":
        return ROAD
    if name == "caution_zone":
        return CAUTION
    return None  # sidewalk etc.


def map_label(label: str):
    """Old single-label mapping (kept for convert_aihub.py --inspect)."""
    cls = map_box_label(label)
    if cls == TRAFFIC_LIGHT:
        return None
    return cls if cls is not None else map_surface_label(label)
