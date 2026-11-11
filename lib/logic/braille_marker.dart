import 'dart:math' as math;

/// Where to draw the yellow square in the bottom bar.
class BrailleMarkerState {
  const BrailleMarkerState({required this.x, required this.opacity, required this.remembered});

  /// 0 = left edge, 1 = right edge of the screen.
  final double x;

  /// 1 = fully visible; goes down to 0 while a remembered marker fades out.
  final double opacity;

  /// True when the braille block is no longer on screen and the marker comes from memory.
  final bool remembered;
}

/// Keeps the bottom-bar marker after the braille block leaves the screen.
///
/// While the block is visible its direction is stored as a world bearing (camera heading from the
/// gyroscope + the block's angle in the image). When the user looks away, the marker stays where the
/// block is: inside the screen if that direction is still in view, otherwise pinned to the left or
/// right edge. It disappears when the user looks back at that direction and the block is not there
/// (the block ended), or after [keep] has passed since the block was last seen.
class BrailleMarker {
  static const keep = Duration(seconds: 8);
  static const fadeFrom = Duration(seconds: 5);
  static const goneInView = Duration(seconds: 1);

  double? _bearingDeg; // right = positive
  DateTime? _lastSeen;
  DateTime? _inViewMissingSince;

  void reset() {
    _bearingDeg = null;
    _lastSeen = null;
    _inViewMissingSince = null;
  }

  /// Angle of a screen x position from the screen center (right = positive), in degrees.
  static double angleOfX(double x, double hfovDeg) {
    final half = hfovDeg * math.pi / 360;
    return math.atan((x - 0.5) * 2 * math.tan(half)) * 180 / math.pi;
  }

  static double xOfAngle(double deg, double hfovDeg) {
    final half = hfovDeg * math.pi / 360;
    return 0.5 + math.tan(deg * math.pi / 180) / (2 * math.tan(half));
  }

  /// [visibleX]: x of the braille path closest to the feet in this frame, or null if none is detected.
  /// [headingDeg]: camera heading from the gyroscope (turning right = increasing).
  BrailleMarkerState? update({
    required double? visibleX,
    required double headingDeg,
    required double hfovDeg,
    required DateTime now,
  }) {
    if (visibleX != null) {
      _bearingDeg = headingDeg + angleOfX(visibleX, hfovDeg);
      _lastSeen = now;
      _inViewMissingSince = null;
      return BrailleMarkerState(x: visibleX, opacity: 1, remembered: false);
    }

    final bearing = _bearingDeg;
    final seen = _lastSeen;
    if (bearing == null || seen == null) return null;
    final age = now.difference(seen);
    if (age >= keep) {
      reset();
      return null;
    }

    final rel = _wrap(bearing - headingDeg);
    final half = hfovDeg / 2;
    if (rel.abs() < half * 0.8) {
      // Looking toward where the block was, but it is not detected: it has ended.
      _inViewMissingSince ??= now;
      if (now.difference(_inViewMissingSince!) >= goneInView) {
        reset();
        return null;
      }
    } else {
      _inViewMissingSince = null;
    }

    final x = rel.abs() >= half ? (rel > 0 ? 1.0 : 0.0) : xOfAngle(rel, hfovDeg).clamp(0.0, 1.0).toDouble();
    final opacity = age <= fadeFrom
        ? 1.0
        : 1.0 - (age - fadeFrom).inMilliseconds / (keep - fadeFrom).inMilliseconds;
    return BrailleMarkerState(x: x, opacity: opacity.clamp(0.0, 1.0).toDouble(), remembered: true);
  }

  static double _wrap(double deg) {
    var a = deg % 360; // Dart: always 0..360 for a positive divisor
    if (a > 180) a -= 360;
    return a;
  }
}
