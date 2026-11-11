/// Voice events produced by the engine. Each fires once per situation (with cooldowns).
enum GuideEvent {
  obstacleOnPath, // "경로에 장애물이 있습니다"
  brailleLost, // "점자블록이 끊겼습니다"
  roadAhead, // "차도입니다"
  crosswalk, // "횡단보도입니다"
  signalRed, // "빨간불입니다"
  signalGreen, // "초록불입니다"
  caution, // "발밑을 조심하세요"
}
