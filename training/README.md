# 모델 학습 (AI Hub 인도보행 영상 → 앱용 tflite)

클래스는 3개입니다: `braille_block`(점자블록), `person`(사람), `obstacle`(장애물 28종 통합).

## 0. 준비 (한 번만)

```powershell
cd C:\ai_project\walk_guide\training
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

NVIDIA GPU가 있으면 PyTorch를 CUDA 버전으로 설치해야 빨라요 (https://pytorch.org 에서 명령 확인).
GPU가 없으면 `--device cpu` 로 돌리되, 시간이 오래 걸리니 `--fraction 0.2` 처럼 일부만 쓰세요.

## 1~3. 데이터 변환 (이 PC 기준 경로)

압축은 이미 풀려 있습니다: `C:\Users\ROG\Downloads\인도보행 영상\` (바운딩박스\Bbox_10_new, Bbox_20_new / 서피스마스킹\Surface_1).
XML 형식은 확인 완료(CVAT XML, 1920x1080, 점자블록 = `braille_guide_blocks` 폴리곤).

```powershell
cd C:\ai_project\walk_guide\training
python convert_aihub.py --bbox "C:\Users\ROG\Downloads\인도보행 영상\바운딩박스" --surface "C:\Users\ROG\Downloads\인도보행 영상\서피스마스킹" --out C:\ai_project\yolo_dataset
```

- 경로에 한글·공백이 있으니 **따옴표 필수**.
- 결과 폴더는 영문 경로(`C:\ai_project\yolo_dataset`)로 둡니다. 같은 C: 드라이브라 이미지는 하드링크로 연결되어 용량을 거의 안 씁니다.
- 끝나면 클래스별 개수가 나옵니다. 점자블록이 3,000개 미만이면 Surface 파일을 더 받으세요.
- 압축 풀린 원본 zip(약 30GB)은 지워도 됩니다.

## 3-2. 10개 클래스 데이터 (v3: 차도·횡단보도·신호등·위험 구역·훼손 점자블록·볼라드 추가)

클래스: braille_block, person, obstacle, bollard, road, crosswalk, caution_zone, braille_damaged,
traffic_light_red, traffic_light_green (`common.py`).

```
python convert_aihub.py --bbox "C:\Users\ROG\Downloads\인도보행 영상\바운딩박스" --surface "C:\Users\ROG\Downloads\인도보행 영상\서피스마스킹" --out C:\ai_project\yolo_v3
python train_all.py --data C:\ai_project\yolo_v3\data.yaml --device 0
```

- 바닥 클래스(점자블록·차도·횡단보도·위험 구역·훼손 점자블록)는 폴리곤을 12조각으로 잘라 조각 박스로 만듭니다.
- 신호등은 세로형(보행자 신호등)만 쓰고 가로형(차량 신호등)은 뺍니다. 색은 박스 안의 켜진 빨강/초록 픽셀로 자동 판정합니다(노란불 제외). 결과 확인용 `yolo_v3\traffic_light_check.jpg`.
- 볼라드가 있는 학습 사진은 2번 넣습니다 (`--bollard-repeat`).
- 차도만 있는 서피스 사진은 30%만 씁니다 (`--road-only-ratio`).
- `train_all.py` = 1차 학습(12 epoch) → 빠진 라벨 채우기(pseudo_label) → 2차 학습(20 epoch). 중간에 멈추면 `--start pseudo` 또는 `--start 2`.

## 3-3. v4: 바운딩박스 8개 + 희귀 클래스 반복

바운딩박스 Bbox_3·7·10·13·17·20·24·28 사용. 신호등 사진 3번, 초록불 사진 5번, 볼라드 사진 2번 넣고,
훼손된 점자블록은 일반 점자블록에 합칩니다(`--keep-damaged` 로 끔). 빠진 라벨 채우기는 효과가 작아서 생략하고
v3 모델에서 이어서 한 번만 학습합니다.

```
python convert_aihub.py --bbox "C:\Users\ROG\Downloads\인도보행 영상\바운딩박스" --surface "C:\Users\ROG\Downloads\인도보행 영상\서피스마스킹" --out C:\ai_project\yolo_v4
python train.py --data C:\ai_project\yolo_v4\data.yaml --device 0 --model runs\walk_guide_v3\weights\best.pt --epochs 20 --name walk_guide_v4
```

## 4. 학습

```powershell
python train.py --data C:\ai_project\yolo_dataset\data.yaml --device 0
```

- 결과: `training\weights\walk_guide.pt`
- 그래프/지표: `training\runs\walk_guide\` (results.png, confusion_matrix.png → 발표 자료로 사용)
- 빠르게 시험만: `--epochs 10 --fraction 0.1`

## 5. (권장) 빠진 라벨 채우고 재학습

바운딩박스 이미지에는 점자블록 라벨이, 서피스마스킹 이미지에는 사람/장애물 라벨이 없습니다.
1차 모델로 빠진 쪽만 채워서 다시 학습하면 성능이 좋아집니다.

```powershell
python pseudo_label.py --dataset C:\ai_project\yolo_dataset --weights weights\walk_guide.pt --device 0
python train.py --data C:\ai_project\yolo_dataset\data.yaml --device 0 --name walk_guide_round2
```

## 6. 앱용 tflite 로 변환

Ultralytics의 LiteRT 변환은 **Windows에서 안 됩니다** (Linux/macOS 전용). 방법 중 하나를 고르세요.

- **가장 쉬움**: `weights\walk_guide.pt` 가 생기면 Claude에게 "변환해줘"라고 하세요. 이 폴더가 연결되어 있으면 대신 변환해서 `assets\models\walk_guide.tflite` 로 넣어 드립니다.
- **WSL(Ubuntu)**:
  ```bash
  cd /mnt/c/ai_project/walk_guide/training
  python3 -m venv .venv-wsl && source .venv-wsl/bin/activate
  pip install "ultralytics[export-litert]"
  python export_tflite.py
  ```

결과가 `assets\models\walk_guide.tflite` 에 저장되면 앱을 다시 실행하세요. 앱 왼쪽 위에 `모델: 학습 모델` 로 표시됩니다.
