"""Run the whole training in one go (leave it running, e.g. overnight):

  1) round 1: train on the converted data (starts from the previous walk_guide model)
  2) pseudo labels: round-1 model fills in the classes each AI Hub subset never labeled
     (roads/braille in 바운딩박스 photos, people/obstacles/traffic lights in 서피스마스킹 photos)
  3) round 2: train again on the completed labels -> weights/walk_guide.pt

Example:
    python train_all.py --data C:\\ai_project\\yolo_v3\\data.yaml --device 0

If it stops in the middle, continue from a step with --start pseudo or --start 2.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(args: list[str]) -> None:
    print("\n>>> " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="convert_aihub.py 가 만든 data.yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--start", choices=["1", "pseudo", "2"], default="1", help="이 단계부터 시작")
    ap.add_argument("--base", default=str(HERE / "runs" / "walk_guide_slices" / "weights" / "best.pt"),
                    help="1단계 시작 가중치 (지난번 모델)")
    ap.add_argument("--epochs1", type=int, default=12)
    ap.add_argument("--epochs2", type=int, default=20)
    args = ap.parse_args()

    data = Path(args.data)
    round1 = HERE / "runs" / "walk_guide_v3_r1" / "weights" / "best.pt"
    steps = ["1", "pseudo", "2"]
    todo = steps[steps.index(args.start):]

    if "1" in todo:
        run(["train.py", "--data", str(data), "--device", args.device, "--model", args.base,
             "--epochs", str(args.epochs1), "--name", "walk_guide_v3_r1"])
    if "pseudo" in todo:
        run(["pseudo_label.py", "--dataset", str(data.parent), "--weights", str(round1),
             "--device", args.device])
    if "2" in todo:
        # Fresh label cache: the pseudo labels changed the label files.
        for cache in data.parent.joinpath("labels").glob("*.cache"):
            cache.unlink()
        run(["train.py", "--data", str(data), "--device", args.device, "--model", str(round1),
             "--epochs", str(args.epochs2), "--name", "walk_guide_v3"])

    print("\n전체 학습 완료: weights\\walk_guide.pt  ->  Claude에게 '변환해줘'라고 하세요.")


if __name__ == "__main__":
    main()
