"""Fill in the labels each AI Hub subset is missing, using a first-round model (pseudo-labeling).

Why: 바운딩박스 images show braille blocks, roads, crosswalks... that were never labeled, and
서피스마스킹 images show people/obstacles/traffic lights that were never labeled. Training on them
as-is teaches the model that those things are background. This script adds confident predictions
for the missing classes only:
  - B_* images (바운딩박스): add surface classes (braille, road, crosswalk, caution_zone, damaged braille)
  - S_* images (서피스마스킹): add box classes (person, obstacle, bollard, traffic lights)
Original labels are backed up once to labels_orig/.

Example:
    python pseudo_label.py --dataset D:\\aihub\\yolo_dataset --weights weights\\walk_guide.pt
    python train.py --data D:\\aihub\\yolo_dataset\\data.yaml --name walk_guide_round2
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from common import BOX_CLASSES, CLASS_NAMES, IMAGE_EXTS, SOURCE_BBOX, SOURCE_SURFACE, SURFACE_CLASSES

MISSING = {SOURCE_BBOX: SURFACE_CLASSES, SOURCE_SURFACE: BOX_CLASSES}


def iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="convert_aihub.py 의 --out 폴더")
    ap.add_argument("--weights", default="weights/walk_guide.pt")
    ap.add_argument("--conf", type=float, default=0.5, help="이 이상 확신하는 예측만 추가")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import YOLO

    ds = Path(args.dataset)
    labels_dir, backup = ds / "labels", ds / "labels_orig"
    if not backup.exists():
        shutil.copytree(labels_dir, backup)
        print(f"원본 라벨 백업: {backup}")
    else:
        # Always rebuild from the originals so running twice does not stack pseudo labels.
        shutil.rmtree(labels_dir)
        shutil.copytree(backup, labels_dir)
        print("원본 라벨에서 다시 시작합니다.")

    model = YOLO(args.weights)
    added = Counter()
    for split in ("train", "val"):
        images = sorted(p for p in (ds / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        print(f"{split}: {len(images):,}장 예측 중...")
        for i in range(0, len(images), args.batch):
            chunk = images[i:i + args.batch]
            results = model.predict([str(p) for p in chunk], conf=args.conf, imgsz=args.imgsz,
                                    device=args.device, verbose=False)
            for img, res in zip(chunk, results):
                source = img.name.split("_", 1)[0]
                wanted = MISSING.get(source)
                if not wanted or res.boxes is None:
                    continue
                label_path = labels_dir / split / f"{img.stem}.txt"
                text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
                lines = [ln for ln in text.splitlines() if ln.strip()]
                existing = [tuple(map(float, ln.split()[1:5])) for ln in lines]
                new_lines = []
                for cls, xywhn in zip(res.boxes.cls.tolist(), res.boxes.xywhn.tolist()):
                    cls = int(cls)
                    if cls not in wanted:
                        continue
                    if any(iou_xywh(xywhn, e) > 0.5 for e in existing):
                        continue
                    new_lines.append(f"{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
                    added[f"{split}/{CLASS_NAMES[cls]}"] += 1
                if new_lines:
                    label_path.write_text("\n".join(lines + new_lines) + "\n", encoding="utf-8")
            if (i // args.batch) % 50 == 0:
                print(f"  {min(i + args.batch, len(images)):,}/{len(images):,}")

    print("\n추가된 의사 라벨:")
    for k in sorted(added):
        print(f"  {k:<24} {added[k]:,}")
    print("\n이제 같은 data.yaml 로 train.py 를 다시 실행하세요.")


if __name__ == "__main__":
    main()
