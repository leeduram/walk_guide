"""Export the trained model to the LiteRT (.tflite) format the Flutter app loads.

Uses the same settings as the official Ultralytics Flutter plugin assets:
format=litert, quantize=w8a32 (int8 weights + fp32 activations, no calibration data needed, GPU friendly).

NOTE: Ultralytics LiteRT export only runs on Linux x86_64 or macOS.
      On Windows, run this inside WSL (Ubuntu), or send training/weights/walk_guide.pt to your helper to export.

Example:
    python export_tflite.py                       # weights/walk_guide.pt -> ../assets/models/walk_guide.tflite
    python export_tflite.py --weights path/to/best.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_MODEL = HERE.parent / "assets" / "models" / "walk_guide.tflite"


def read_metadata(path: Path) -> dict | None:
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith("metadata.json"):
                    return json.loads(zf.read(name))
    except zipfile.BadZipFile:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "weights" / "walk_guide.pt"))
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--out", default=str(APP_MODEL))
    args = ap.parse_args()

    if platform.system() == "Windows":
        sys.exit("LiteRT 변환은 Windows에서 지원되지 않습니다. WSL(Ubuntu)에서 실행하거나 README의 방법을 보세요.")

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"가중치 파일이 없습니다: {weights}")

    from ultralytics import YOLO

    exported = YOLO(str(weights)).export(format="litert", quantize="w8a32", nms=None, imgsz=args.imgsz, batch=1)
    exported = Path(exported)
    if exported.is_dir():
        found = sorted(exported.rglob("*.tflite"))
        if not found:
            sys.exit(f"변환 결과에서 .tflite 를 찾지 못했습니다: {exported}")
        exported = found[0]

    meta = read_metadata(exported)
    if meta:
        print(f"metadata: task={meta.get('task')} imgsz={meta.get('imgsz')} names={meta.get('names')}")
    else:
        print("[주의] tflite 안에 metadata.json 이 없습니다. 앱에서 클래스 이름을 못 읽을 수 있습니다.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, out)
    print(f"\n앱 모델 저장 완료: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("Android Studio에서 앱을 다시 실행하면 이 모델을 사용합니다.")


if __name__ == "__main__":
    main()
