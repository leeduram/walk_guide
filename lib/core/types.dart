import 'dart:ui';

/// What the app does with each detection.
enum ObjectGroup {
  braille, // braille block piece
  brailleDamaged, // damaged braille block piece (still a guide path, drawn in another colour)
  person,
  obstacle, // includes bollards
  road, // roadway / alley / bike lane piece
  crosswalk, // crosswalk piece
  caution, // caution zone piece under foot: stairs, manhole, grating, tree pit, repair zone
  trafficRed, // traffic light showing red
  trafficGreen, // traffic light showing green
}

extension ObjectGroupLabel on ObjectGroup {
  String get korean => switch (this) {
        ObjectGroup.braille => '점자블록',
        ObjectGroup.brailleDamaged => '훼손된 점자블록',
        ObjectGroup.person => '사람',
        ObjectGroup.obstacle => '장애물',
        ObjectGroup.road => '차도',
        ObjectGroup.crosswalk => '횡단보도',
        ObjectGroup.caution => '위험 구역',
        ObjectGroup.trafficRed => '빨간불',
        ObjectGroup.trafficGreen => '초록불',
      };

  /// Floor surfaces detected as slice pieces (smoothed and drawn as bands).
  bool get isSurface =>
      this == ObjectGroup.braille ||
      this == ObjectGroup.brailleDamaged ||
      this == ObjectGroup.road ||
      this == ObjectGroup.crosswalk ||
      this == ObjectGroup.caution;
}

/// One detection from the model, already converted to the app's coordinate systems.
class Detection {
  const Detection({
    required this.group,
    required this.label,
    required this.confidence,
    required this.imageBox,
    required this.viewBox,
  });

  final ObjectGroup group;

  /// Raw class name from the model (e.g. "obstacle", or "car" for the COCO fallback model).
  final String label;
  final double confidence;

  /// Box normalized to the camera image (0..1). Used for distance geometry.
  final Rect imageBox;

  /// Box normalized to the visible screen (0..1). Used for RoI logic and drawing.
  final Rect viewBox;
}

/// Maps model class names to [ObjectGroup].
class ClassMapper {
  // Custom model trained with training/convert_aihub.py.
  static const _braille = {'braille_block', 'braille', 'braille_guide_blocks'};
  static const _person = {'person'};
  static const _obstacle = {'obstacle', 'bollard'}; // bollard is trained separately, used as an obstacle
  static const _surfaces = {
    'braille_damaged': ObjectGroup.brailleDamaged,
    // 'road' is detected by the model but not used (road-entry warning removed).
    'crosswalk': ObjectGroup.crosswalk,
    // 'caution_zone' (stairs / manhole / grating ...) is detected by the model but not used:
    // stairs had too few training samples, so the whole feature was removed.
    'traffic_light_red': ObjectGroup.trafficRed,
    'traffic_light_green': ObjectGroup.trafficGreen,
  };

  // Used only with the COCO fallback model (before the custom model is trained).
  static const _cocoObstacles = {
    'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'dog', 'cat', 'chair', 'potted plant', 'suitcase', 'traffic light', 'umbrella',
    'dining table', 'backpack', 'handbag',
  };

  static ObjectGroup? map(String className) {
    final name = className.trim().toLowerCase();
    if (_braille.contains(name)) return ObjectGroup.braille;
    final other = _surfaces[name];
    if (other != null) return other;
    if (_person.contains(name)) return ObjectGroup.person;
    if (_obstacle.contains(name) || _cocoObstacles.contains(name)) return ObjectGroup.obstacle;
    return null;
  }
}
