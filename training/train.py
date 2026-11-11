"""Fine-tune a YOLO nano detector on the converted dataset.

Example:
    python train.py --data D:\\aihub\\yolo_dataset\\data.yaml
    python train.py --data D:\\aihub\\yolo_dataset\\data.yaml --epochs 80 --batch 32 --device 0

The best checkpoint is copied to training/weights/walk_guide.pt.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="convert_aihub.py 가 만든 data.yaml")
    ap.add_argument("--model", default="yolo26n.pt", help="시작 가중치 (yolo26n.pt 권장, 안되면 yolo11n.pt)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="GPU 메모리가 부족하면 8로")
    ap.add_argument("--device", default=None, help="0 = 첫 번째 NVIDIA GPU, cpu = CPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="walk_guide")
    ap.add_argument("--fraction", type=float, default=1.0, help="학습 데이터 중 사용할 비율 (빠른 시험용 0.1)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(HERE / "runs"),
        name=args.name,
        exist_ok=True,
        fraction=args.fraction,
        resume=args.resume,
        patience=15,
        # 점자블록은 노란색이 핵심 단서라 색조(hue) 변형을 거의 주지 않는다.
        hsv_h=0.005,
        fliplr=0.5,
        flipud=0.0,
        plots=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        print(f"best.pt 를 찾지 못했습니다: {best}")
        return
    out = HERE / "weights" / "walk_guide.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy2(best, out)
    print(f"\n학습 완료: {out}")
    print("다음 단계: python export_tflite.py  (Windows에서는 WSL 또는 아래 README 참고)")


if __name__ == "__main__":
    main()
