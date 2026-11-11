/// Parameters. There is no settings screen, so these defaults are what the app always uses:
/// - cameraHeightM 1.3: phone held in front of the chest in landscape (adult chest height).
/// - verticalFovDeg 52: Samsung main camera (26 mm equivalent) 4:3 frame, short side, in landscape.
/// - dangerDistanceM 3 / dangerAreaRatio 0.15: the 3 m rule and the 15% box-area rule from the spec.
/// - maxDistanceM 5: objects farther than this are ignored.
/// - confidence 0.45 (people/obstacles), brailleConfidence 0.25 (braille pieces are small),
///   surfaceConfidence 0.25 (crosswalk pieces), trafficConfidence 0.3 (pedestrian lights).
class GuideSettings {
  GuideSettings({
    this.cameraHeightM = 1.3,
    this.verticalFovDeg = 52.0,
    this.pitchOffsetDeg = 0.0,
    this.dangerDistanceM = 3.0,
    this.maxDistanceM = 5.0,
    this.dangerAreaRatio = 0.15,
    this.confidence = 0.45,
    this.brailleConfidence = 0.25,
    this.surfaceConfidence = 0.25,
    this.trafficConfidence = 0.3,
    this.beepEnabled = true,
    this.voiceEnabled = true,
  });

  /// Height of the phone camera above the ground (meters).
  double cameraHeightM;

  /// Vertical field of view of the camera image as held (landscape, degrees).
  double verticalFovDeg;

  /// Added to the measured camera pitch, to correct a systematic distance error.
  double pitchOffsetDeg;

  /// Objects closer than this are dangerous (highlight + beep).
  double dangerDistanceM;

  /// Objects farther than this are ignored completely.
  double maxDistanceM;

  /// Fallback rule from the original spec: a box covering this share of the screen is dangerous
  /// even when the ground distance cannot be computed (e.g. a sign hanging at head height).
  double dangerAreaRatio;

  /// Minimum detector confidence for people and obstacles.
  double confidence;

  /// Minimum detector confidence for braille blocks (lower: they are harder to detect).
  double brailleConfidence;

  /// Minimum detector confidence for road / crosswalk / caution-zone pieces.
  double surfaceConfidence;

  /// Minimum detector confidence for pedestrian traffic lights (small objects: lower).
  double trafficConfidence;

  bool beepEnabled;
  bool voiceEnabled;
}
