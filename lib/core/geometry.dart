import 'dart:math' as math;
import 'dart:ui';

/// Converts boxes from camera-image coordinates to screen coordinates.
///
/// The camera preview is drawn "fill center": scaled until it covers the whole screen, then the
/// overflow is cropped equally on both sides. Boxes must go through the same transform so they line
/// up with what the user sees.
class ViewMapper {
  /// [imageAspect] = image width / image height (a 4:3 camera frame in portrait is 0.75).
  static Rect imageToView(Rect n, double imageAspect, Size view) {
    if (view.width <= 0 || view.height <= 0 || imageAspect <= 0) return n;
    final scale = math.max(view.width / imageAspect, view.height / 1.0);
    final dispW = imageAspect * scale;
    final dispH = 1.0 * scale;
    final offX = (view.width - dispW) / 2;
    final offY = (view.height - dispH) / 2;
    return Rect.fromLTRB(
      (offX + n.left * dispW) / view.width,
      (offY + n.top * dispH) / view.height,
      (offX + n.right * dispW) / view.width,
      (offY + n.bottom * dispH) / view.height,
    );
  }
}

/// Monocular distance from ground geometry: no LiDAR or ultrasonic sensor.
///
/// A point on flat ground seen [angle] below the horizon by a camera [h] meters high is
/// h / tan(angle) meters away. The angle is the phone pitch (accelerometer) plus the angle of the
/// image row relative to the optical axis.
class GroundDistance {
  static double? estimate({
    required double yNorm,
    required double pitchDeg,
    required double cameraHeightM,
    required double verticalFovDeg,
  }) {
    final halfFov = verticalFovDeg * math.pi / 360.0;
    final rowAngle = math.atan((yNorm - 0.5) * 2.0 * math.tan(halfFov));
    final angle = pitchDeg * math.pi / 180.0 + rowAngle;
    if (angle < math.pi / 180.0) return null; // at or above the horizon: not a ground point
    return cameraHeightM / math.tan(angle);
  }

  /// Camera pitch below the horizon (degrees) from an accelerometer reading.
  /// Upright phone -> 0, lying flat screen-up (camera facing down) -> 90.
  /// Uses the in-screen gravity magnitude, so it works the same in portrait and both landscapes.
  static double pitchFromAccelerometer(double x, double y, double z) =>
      math.atan2(z, math.sqrt(x * x + y * y)) * 180.0 / math.pi;
}

class RectMath {
  static double area(Rect r) => math.max(0.0, r.width) * math.max(0.0, r.height);

  static double intersection(Rect a, Rect b) {
    final w = math.min(a.right, b.right) - math.max(a.left, b.left);
    final h = math.min(a.bottom, b.bottom) - math.max(a.top, b.top);
    return (w <= 0 || h <= 0) ? 0.0 : w * h;
  }

  static double iou(Rect a, Rect b) {
    final inter = intersection(a, b);
    final union = area(a) + area(b) - inter;
    return union <= 0 ? 0.0 : inter / union;
  }

  /// Share of [a] covered by [b] ("intersection over area").
  static double ioa(Rect a, Rect b) {
    final ar = area(a);
    return ar <= 0 ? 0.0 : intersection(a, b) / ar;
  }

  static Rect lerp(Rect from, Rect to, double t) => Rect.fromLTRB(
        from.left + (to.left - from.left) * t,
        from.top + (to.top - from.top) * t,
        from.right + (to.right - from.right) * t,
        from.bottom + (to.bottom - from.bottom) * t,
      );
}
