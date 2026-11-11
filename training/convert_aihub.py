"""Convert AI Hub 인도보행 영상 (CVAT XML) into the walk_guide YOLO detection dataset.

Classes (see common.py): braille_block, person, obstacle, bollard, road, crosswalk, caution_zone,
braille_damaged, traffic_light_red, traffic_light_green

Example (Windows cmd):
    python convert_aihub.py --bbox "C:\\...\\바운딩박스" --surface "C:\\...\\서피스마스킹" --out C:\\ai_project\\yolo_v3

What it does
- 바운딩박스 set (<box>): person, bollard, obstacle (other 27 types merged), pedestrian traffic lights.
  Only vertical (pedestrian) signals are kept; horizontal vehicle signals are dropped. The colour is read
  from the pixels (lit red / lit green, amber excluded); unclear ones are dropped.
  A check sheet of the colour decisions is written to <out>/traffic_light_check.jpg.
- 서피스마스킹 set (<polygon> + attribute): braille_block / braille_damaged, road (roadway, alley,
  bike_lane), crosswalk (roadway/alley with attribute crosswalk), caution_zone (stairs, manhole,
  grating, tree_zone, repair_zone). Each polygon class is cut into horizontal slices (default 12 per
  image height) and every slice gets its own box, so the detector learns pieces that follow the shape.
- 서피스마스킹 images are kept when they contain braille / crosswalk / caution / damaged braille;
  images with only road are kept at --road-only-ratio (they are very common).
- Rare classes are repeated in training: bollard x2, traffic light x3, green light x5.
- Damaged braille is merged into braille_block by default (too few samples to learn); --keep-damaged.
- Train/val split is done per XML file (one XML = one recording sequence) so near-identical
  frames do not leak from train into val.
- Images are hard-linked when possible (no extra disk space), otherwise copied.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from common import (BOLLARD, BOX_CLASSES, BRAILLE, BRAILLE_DAMAGED, CAUTION, CLASS_NAMES, CROSSWALK,
                    IMAGE_EXTS, ROAD, SOURCE_BBOX, SOURCE_SURFACE, SURFACE_CLASSES, TL_GREEN, TL_RED,
                    TRAFFIC_LIGHT, map_box_label, map_label, map_surface_label)

MIN_BOX_PX = 8          # drop boxes smaller than this (width or height, in pixels)
MIN_BOX_AREA = 0.0004   # drop boxes smaller than 0.04% of the image


# --------------------------------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------------------------------
SKIP_DIRS = {"mask"}  # 서피스마스킹 MASK/*.png are label masks, not photos


def find_files(root: Path, exts: set[str]) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]
        for fn in filenames:
            if Path(fn).suffix.lower() in exts:
                out.append(Path(dirpath) / fn)
    return sorted(out)


def build_image_index(roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        for p in find_files(root, IMAGE_EXTS):
            index[p.name.lower()].append(p)
    return index


def resolve_image(name: str, xml_path: Path, index: dict[str, list[Path]]) -> Path | None:
    rel = Path(name.replace("\\", "/"))
    direct = xml_path.parent / rel
    if direct.is_file():
        return direct
    candidates = index.get(rel.name.lower(), [])
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        # Several images share a file name: prefer the one closest to the XML file.
        xml_parts = xml_path.parent.parts
        return max(candidates, key=lambda c: sum(1 for a, b in zip(c.parts, xml_parts) if a == b))
    return None


# --------------------------------------------------------------------------------------------------
# XML parsing (CVAT "for images" format)
# --------------------------------------------------------------------------------------------------
def parse_points(points: str) -> list[tuple[float, float]]:
    pts = []
    for pair in points.replace(" ", "").split(";"):
        if not pair:
            continue
        x, y = pair.split(",")[:2]
        pts.append((float(x), float(y)))
    return pts


def iter_images(xml_path: Path):
    """Yield (image_name, width, height, boxes, polygons) per <image> element.

    boxes:    [(class_id or TRAFFIC_LIGHT, x1, y1, x2, y2), ...] from <box>
    polygons: {class_id: [[(x, y), ...], ...]} from <polygon>/<polyline> (surface classes)
    """
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "image":
            continue
        name = elem.get("name", "")
        w = float(elem.get("width", 0) or 0)
        h = float(elem.get("height", 0) or 0)
        boxes = []
        polygons: dict[int, list] = defaultdict(list)
        for child in elem:
            label = child.get("label", "")
            if child.tag == "box":
                cls = map_box_label(label)
                if cls is None:
                    continue
                x1, y1 = float(child.get("xtl")), float(child.get("ytl"))
                x2, y2 = float(child.get("xbr")), float(child.get("ybr"))
                boxes.append((cls, x1, y1, x2, y2))
            elif child.tag in ("polygon", "polyline"):
                attr_el = child.find("attribute")
                attr = attr_el.text if attr_el is not None else ""
                cls = map_surface_label(label, attr or "")
                if cls is None:
                    continue
                pts = parse_points(child.get("points", ""))
                if len(pts) >= 3:
                    polygons[cls].append(pts)
        yield name, w, h, boxes, dict(polygons)
        elem.clear()


def slice_polygons(polys, w: float, h: float, n_slices: int, cls: int, scale: float = 0.5):
    """Cut the union of the polygons into horizontal slices; return one (cls, x1, y1, x2, y2) box per
    connected run of columns inside each slice (two parallel strips give two boxes)."""
    import numpy as np
    from PIL import Image, ImageDraw

    mw, mh = max(1, int(w * scale)), max(1, int(h * scale))
    mask = Image.new("L", (mw, mh), 0)
    draw = ImageDraw.Draw(mask)
    for pts in polys:
        draw.polygon([(x * scale, y * scale) for x, y in pts], fill=1)
    m = np.asarray(mask, dtype=bool)
    edges = np.linspace(0, mh, n_slices + 1).astype(int)
    boxes = []
    for y0, y1 in zip(edges[:-1], edges[1:]):
        band = m[y0:y1]
        cols = band.any(axis=0)
        if not cols.any():
            continue
        padded = np.concatenate(([False], cols, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
        for xs, xe in zip(starts, ends):
            rows = np.where(band[:, xs:xe].any(axis=1))[0]
            boxes.append((cls, xs / scale, (y0 + rows[0]) / scale, xe / scale, (y0 + rows[-1] + 1) / scale))
    return boxes


def is_pedestrian_signal(box) -> bool:
    """Korean pedestrian signals are vertical (red figure above green figure); vehicle signals are
    horizontal. Only pedestrian signals are kept: a vehicle green can mean a pedestrian red."""
    x1, y1, x2, y2 = box
    return (y2 - y1) >= 1.3 * (x2 - x1)


def traffic_light_color(img, box):
    """TL_RED / TL_GREEN from the lit pixels inside a traffic light box, or None when unclear."""
    import numpy as np

    x1, y1, x2, y2 = box
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    crop = img.crop((int(x1), int(y1), int(x2 + 0.999), int(y2 + 0.999))).convert("HSV")
    a = np.asarray(crop)
    hue, sat, val = a[..., 0].astype(int), a[..., 1], a[..., 2]
    lit = (val >= 150) & (sat >= 110)
    red = int((lit & ((hue <= 14) | (hue >= 235))).sum())    # PIL hue 0..255: <=20 deg or >=332 deg
    green = int((lit & (hue >= 85) & (hue <= 140)).sum())    # 120..198 deg (Korean signals look cyan-green)
    yellow = int((lit & (hue >= 15) & (hue <= 40)).sum())    # amber light: not red
    need = max(4, int(0.01 * hue.size))
    if yellow >= red and yellow >= green:
        return None
    if red >= need and red >= 2 * green:
        return TL_RED
    if green >= need and green >= 2 * red:
        return TL_GREEN
    return None


class LightCheckSheet:
    """Collects a few traffic light crops per decision for a visual check of the colour rule."""

    def __init__(self, per_kind: int = 48):
        self.per_kind = per_kind
        self.tiles: dict[str, list] = {"red": [], "green": [], "unclear": []}

    def add(self, img, box, cls) -> None:
        kind = "red" if cls == TL_RED else "green" if cls == TL_GREEN else "unclear"
        if len(self.tiles[kind]) >= self.per_kind or box[2] - box[0] < 10 or box[3] - box[1] < 10:
            return
        crop = img.crop((int(box[0]), int(box[1]), int(box[2] + 0.999), int(box[3] + 0.999)))
        crop.thumbnail((72, 72))
        self.tiles[kind].append(crop.copy())

    def save(self, path: Path) -> None:
        from PIL import Image, ImageDraw

        cols, tile = 16, 80
        rows = sum((len(v) + cols - 1) // cols for v in self.tiles.values()) + len(self.tiles)
        sheet = Image.new("RGB", (cols * tile, max(1, rows) * tile), (40, 40, 40))
        draw = ImageDraw.Draw(sheet)
        colors = {"red": (255, 60, 60), "green": (60, 255, 120), "unclear": (180, 180, 180)}
        r = 0
        for kind, tiles in self.tiles.items():
            draw.text((6, r * tile + 30), f"{kind} ({len(tiles)})", fill=colors[kind])
            r += 1
            for i, t in enumerate(tiles):
                x, y = (i % cols) * tile, (r + i // cols) * tile
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=colors[kind], width=3)
                sheet.paste(t, (x + (tile - t.width) // 2, y + (tile - t.height) // 2))
            r += (len(tiles) + cols - 1) // cols
        sheet.save(path, quality=90)


def to_yolo_lines(objs, w: float, h: float, allowed: set[int]) -> list[str]:
    lines = []
    for cls, x1, y1, x2, y2 in objs:
        if cls not in allowed:
            continue
        x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX or (bw * bh) / (w * h) < MIN_BOX_AREA:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


# --------------------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------------------
def is_val(xml_path: Path, val_ratio: float) -> bool:
    digest = hashlib.md5(str(xml_path).encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10000) / 10000.0 < val_ratio


def place_image(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists():
        return
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def safe_stem(src: Path, prefix: str) -> str:
    tag = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{tag}_{src.stem}"


# --------------------------------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------------------------------
def keep_fraction(key: str, ratio: float) -> bool:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (int(digest[8:16], 16) % 10000) / 10000.0 < ratio


def convert(args) -> None:
    from PIL import Image

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    jobs = [(Path(r), SOURCE_BBOX) for r in (args.bbox or [])]
    jobs += [(Path(r), SOURCE_SURFACE) for r in (args.surface or [])]
    if not jobs:
        sys.exit("--bbox 또는 --surface 폴더를 하나 이상 지정하세요.")

    stats = {"images": Counter(), "instances": Counter(), "missing": 0, "skipped": 0,
             "xml": 0, "xml_failed": 0, "lights": Counter(), "bollard_repeat": 0}
    sheet = LightCheckSheet()
    must_keep = {BRAILLE, BRAILLE_DAMAGED, CROSSWALK, CAUTION}

    for root, source in jobs:
        if not root.exists():
            print(f"[경고] 폴더 없음: {root}")
            continue
        print(f"\n[{source}] {root} 이미지 목록 만드는 중...")
        index = build_image_index([root])
        xml_files = find_files(root, {".xml"})
        print(f"  이미지 {sum(len(v) for v in index.values()):,}장, XML {len(xml_files):,}개")
        if args.max_images:
            random.Random(0).shuffle(xml_files)
        taken = 0
        for xml_path in xml_files:
            if args.max_images and taken >= args.max_images:
                break
            stats["xml"] += 1
            split = "val" if is_val(xml_path, args.val_ratio) else "train"
            try:
                for name, w, h, boxes, polygons in iter_images(xml_path):
                    if args.max_images and taken >= args.max_images:
                        break
                    src = resolve_image(name, xml_path, index)
                    if src is None or w <= 0 or h <= 0:
                        stats["missing"] += 1
                        continue

                    objs = []
                    if source == SOURCE_SURFACE:
                        present = set(polygons)
                        if not (present & must_keep):
                            if ROAD not in present or not keep_fraction(str(src), args.road_only_ratio):
                                stats["skipped"] += 1
                                continue
                        if args.merge_damaged and BRAILLE_DAMAGED in polygons:
                            polygons.setdefault(BRAILLE, []).extend(polygons.pop(BRAILLE_DAMAGED))
                        for cls, polys in polygons.items():
                            if args.slices > 0:
                                objs += slice_polygons(polys, w, h, args.slices, cls)
                            else:
                                xs = [x for p in polys for x, _ in p]
                                ys = [y for p in polys for _, y in p]
                                objs.append((cls, min(xs), min(ys), max(xs), max(ys)))
                        allowed = SURFACE_CLASSES
                    else:
                        img = None
                        for cls, x1, y1, x2, y2 in boxes:
                            if cls == TRAFFIC_LIGHT:
                                if not is_pedestrian_signal((x1, y1, x2, y2)):
                                    stats["lights"]["vehicle(dropped)"] += 1
                                    continue
                                if img is None:
                                    img = Image.open(src).convert("RGB")
                                color = traffic_light_color(img, (x1, y1, x2, y2))
                                sheet.add(img, (x1, y1, x2, y2), color)
                                stats["lights"]["red" if color == TL_RED else "green" if color == TL_GREEN
                                                else "unclear(dropped)"] += 1
                                if color is None:
                                    continue
                                cls = color
                            objs.append((cls, x1, y1, x2, y2))
                        allowed = BOX_CLASSES

                    lines = to_yolo_lines(objs, w, h, allowed)
                    if source == SOURCE_SURFACE and not lines:
                        stats["skipped"] += 1
                        continue
                    stem = safe_stem(src, source)
                    # Rare classes: put the training image in several times.
                    present_ids = {int(ln.split()[0]) for ln in lines}
                    repeat = 1
                    if split == "train":
                        if BOLLARD in present_ids:
                            repeat = max(repeat, args.bollard_repeat)
                        if present_ids & {TL_RED, TL_GREEN}:
                            repeat = max(repeat, args.light_repeat)
                        if TL_GREEN in present_ids:
                            repeat = max(repeat, args.green_repeat)
                    copies = [stem] + [f"{stem}_r{k}" for k in range(2, repeat + 1)]
                    stats["bollard_repeat"] += len(copies) - 1
                    for st in copies:
                        place_image(src, out / "images" / split / f"{st}{src.suffix.lower()}", args.copy)
                        (out / "labels" / split / f"{st}.txt").write_text(
                            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    stats["images"][f"{split}/{source}"] += 1
                    for ln in lines:
                        stats["instances"][f"{split}/{CLASS_NAMES[int(ln.split()[0])]}"] += 1
                    taken += 1
            except ET.ParseError as e:
                stats["xml_failed"] += 1
                print(f"  [XML 오류] {xml_path}: {e}")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    sheet.save(out / "traffic_light_check.jpg")

    print("\n================ 변환 결과 ================")
    print(f"XML 처리 {stats['xml']:,}개 (실패 {stats['xml_failed']})")
    for k in sorted(stats["images"]):
        print(f"이미지  {k:<16} {stats['images'][k]:>8,}")
    for split in ("train", "val"):
        for name in CLASS_NAMES:
            k = f"{split}/{name}"
            print(f"객체    {k:<28} {stats['instances'].get(k, 0):>8,}")
    print(f"신호등 색 판정: " + ", ".join(f"{k} {v:,}" for k, v in stats["lights"].items()))
    print(f"볼라드·신호등 사진 추가 복사 {stats['bollard_repeat']:,}장")
    print(f"이미지 파일 못 찾음 {stats['missing']:,} / 제외 {stats['skipped']:,}")
    print(f"\ndata.yaml: {data_yaml}")
    print(f"신호등 색 확인용: {out / 'traffic_light_check.jpg'}")


def inspect(root: Path) -> None:
    xml_files = find_files(root, {".xml"})
    images = find_files(root, IMAGE_EXTS)
    print(f"XML {len(xml_files):,}개, 이미지 {len(images):,}장")
    if images:
        print(f"이미지 예시: {images[0]}")
    if not xml_files:
        print("XML이 없습니다. 압축을 다 풀었는지 확인하세요.")
        return
    xml_path = xml_files[0]
    print(f"XML 예시: {xml_path}")
    tags, labels, mapped = Counter(), Counter(), Counter()
    first_image = None
    n_images = 0
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        tags[elem.tag] += 1
        if elem.tag == "image":
            n_images += 1
            if first_image is None:
                first_image = dict(elem.attrib)
            for child in elem:
                lb = child.get("label")
                if lb:
                    labels[(child.tag, lb)] += 1
                    cls = map_label(lb)
                    mapped[CLASS_NAMES[cls] if cls is not None else "(무시)"] += 1
            elem.clear()
    print(f"태그 빈도: {dict(tags.most_common(12))}")
    print(f"<image> {n_images}개, 첫 이미지 속성: {first_image}")
    print("라벨 빈도:")
    for (tag, lb), c in labels.most_common(40):
        print(f"  {tag:<8} {lb:<32} {c}")
    print(f"변환 후 클래스: {dict(mapped)}")
    if first_image:
        index = build_image_index([root])
        found = resolve_image(first_image.get("name", ""), xml_path, index)
        print(f"첫 이미지 파일 찾기: {found if found else '실패'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", metavar="DIR", help="XML 구조와 라벨만 확인하고 끝냄")
    ap.add_argument("--bbox", nargs="*", help="압축 푼 바운딩박스 폴더(들)")
    ap.add_argument("--surface", nargs="*", help="압축 푼 서피스마스킹 폴더(들)")
    ap.add_argument("--out", help="YOLO 데이터셋을 만들 폴더")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--max-images", type=int, default=0, help="폴더 하나당 최대 이미지 수 (0=전부)")
    ap.add_argument("--road-only-ratio", type=float, default=0.3,
                    help="차도만 있는 서피스마스킹 이미지 중 사용할 비율")
    ap.add_argument("--bollard-repeat", type=int, default=2, help="볼라드가 있는 학습 이미지를 몇 번 넣을지")
    ap.add_argument("--light-repeat", type=int, default=3, help="신호등이 있는 학습 이미지를 몇 번 넣을지")
    ap.add_argument("--green-repeat", type=int, default=5, help="초록불이 있는 학습 이미지를 몇 번 넣을지")
    ap.add_argument("--keep-damaged", dest="merge_damaged", action="store_false",
                    help="훼손된 점자블록을 따로 학습 (기본: 데이터가 너무 적어 일반 점자블록에 합침)")
    ap.add_argument("--copy", action="store_true", help="하드링크 대신 항상 복사")
    ap.add_argument("--slices", type=int, default=12,
                    help="바닥 폴리곤(점자블록·차도 등)을 위아래로 자르는 조각 수 (0 = 박스 하나)")
    args = ap.parse_args()

    if args.inspect:
        inspect(Path(args.inspect))
        return
    if not args.out:
        ap.error("--out 을 지정하세요")
    convert(args)


if __name__ == "__main__":
    main()
