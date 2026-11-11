import 'package:flutter/material.dart';

import '../core/types.dart';
import '../logic/braille_marker.dart';
import '../logic/braille_path.dart';
import '../logic/guide_engine.dart';

/// Draws everything on top of the camera preview. All boxes arrive normalized to the screen (0..1).
class GuideOverlayPainter extends CustomPainter {
  GuideOverlayPainter({
    required this.frame,
    required this.blinkOn,
    this.marker,
    this.debugDetections,
  });

  final GuideFrame frame;
  final bool blinkOn; // toggled by the screen for the danger flash
  final BrailleMarkerState? marker; // remembered braille direction for the bottom bar
  final List<Detection>? debugDetections; // TEMPORARY test mode (kShowDetections); null = off

  static const brailleColor = Color(0xFFE5FF00); // fluorescent yellow
  static const markerColor = Color(0xFFFFFF00); // brightest yellow for the bottom direction square
  static const dangerColor = Color(0xFFFF2D2D);
  static const damagedColor = Color(0xFFFF8C00); // orange: damaged braille blocks
  static const cautionColor = Color(0xFFC040FF); // purple: manhole, grating, stairs... in the path
  static const greenColor = Color(0xFF00E676); // green traffic light

  Rect _px(Rect n, Size s) => Rect.fromLTRB(n.left * s.width, n.top * s.height, n.right * s.width, n.bottom * s.height);

  @override
  void paint(Canvas canvas, Size size) {
    final street = frame.street;

    // Stepping onto the road: the road right ahead turns red.
    if (street.roadWarning) {
      _drawBands(canvas, size, street.roadBoxes, dangerColor, 0.35, 0);
    }
    // Caution zones in the walking path (manhole, grating, stairs, tree pit, repair zone): purple.
    _drawBands(canvas, size, street.cautionBoxes, cautionColor, 0.6, 5);

    // 3. Braille blocks: one smooth band per chain of detected pieces, following the strip.
    // Normal blocks are yellow, damaged blocks orange.
    // No outline: an outline is drawn half outside the band and made it look wider than the blocks.
    _drawBands(canvas, size, frame.brailleBoxes, brailleColor, 0.85, 0);
    _drawBands(canvas, size, frame.damagedBrailleBoxes, damagedColor, 0.85, 0);

    // 1 + 2. Only dangerous objects (within 3 m in the RoI) are drawn, as a flashing red box.
    // Everything farther away is not drawn at all.
    for (final t in frame.objects) {
      if (!t.danger) continue;
      final r = _px(t.viewBox, size);
      canvas.drawRect(r, Paint()..color = dangerColor.withValues(alpha: blinkOn ? 0.35 : 0.12));
      canvas.drawRRect(
        RRect.fromRectAndRadius(r, const Radius.circular(6)),
        Paint()
          ..color = dangerColor
          ..style = PaintingStyle.stroke
          ..strokeWidth = 6,
      );
    }

    _drawBottomBar(canvas, size);
    _drawSignal(canvas, size, street.signal);
    final debug = debugDetections;
    if (debug != null) _drawDebug(canvas, size, debug);
  }

  /// TEMPORARY test mode: raw detections as thin boxes with a short class tag, and the zones the
  /// road / crosswalk rules look at.
  void _drawDebug(Canvas canvas, Size size, List<Detection> debugDetections) {
    const tags = {
      ObjectGroup.braille: 'B',
      ObjectGroup.brailleDamaged: 'BD',
      ObjectGroup.person: 'P',
      ObjectGroup.obstacle: 'O',
      ObjectGroup.road: 'R',
      ObjectGroup.crosswalk: 'C',
      ObjectGroup.caution: 'Z',
      ObjectGroup.trafficRed: 'TR',
      ObjectGroup.trafficGreen: 'TG',
    };
    const colors = {
      ObjectGroup.road: Color(0xFFFF6E6E),
      ObjectGroup.crosswalk: Color(0xFF40C4FF),
      ObjectGroup.trafficRed: Color(0xFFFF1744),
      ObjectGroup.trafficGreen: Color(0xFF00E676),
      ObjectGroup.braille: Color(0xFFFFFF00),
    };
    final zone = Paint()
      ..color = Colors.white.withValues(alpha: 0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    canvas.drawRect(_px(const Rect.fromLTRB(0.25, 0.6, 0.75, 1.0), size), zone); // road zone
    canvas.drawRect(_px(const Rect.fromLTRB(0.1, 0.4, 0.9, 1.0), size), zone); // crosswalk zone
    final counts = <String, int>{};
    for (final d in debugDetections) {
      final tag = tags[d.group] ?? '?';
      counts[tag] = (counts[tag] ?? 0) + 1;
      final color = colors[d.group] ?? Colors.white;
      final r = _px(d.viewBox, size);
      canvas.drawRect(
        r,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5,
      );
      final tp = TextPainter(
        text: TextSpan(
          text: '$tag ${(d.confidence * 100).round()}',
          style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w700),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, r.topLeft + const Offset(2, 1));
    }
    final summary = [
      counts.entries.map((e) => '${e.key}:${e.value}').join(' '),
      if (frame.street.crosswalkContext) 'CW-context',
      if (frame.street.roadWarning) 'ROAD',
    ].join('  ');
    final tp = TextPainter(
      text: TextSpan(
        text: summary.isEmpty ? '(no detections)' : summary,
        style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w700, backgroundColor: Colors.black54),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: size.width);
    tp.paint(canvas, const Offset(8, 8));
  }

  /// Pieces joined into smooth bands (see BraillePath), filled with [color] and an optional edge.
  void _drawBands(Canvas canvas, Size size, List<Rect> pieces, Color color, double fillAlpha, double edge) {
    if (pieces.isEmpty) return;
    final fill = Paint()..color = color.withValues(alpha: fillAlpha);
    final stroke = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = edge
      ..strokeJoin = StrokeJoin.round;
    for (final c in BraillePath.chains(pieces)) {
      final path = BraillePath.band(c, size);
      canvas.drawPath(path, fill);
      if (edge > 0) canvas.drawPath(path, stroke);
    }
  }

  /// Traffic light near a crosswalk: a solid red or green square at the top edge, above the light.
  void _drawSignal(Canvas canvas, Size size, TrafficSignal? signal) {
    if (signal == null) return;
    final side = GuideEngine.footZone.height * size.height;
    final x = (signal.box.center.dx * size.width).clamp(side / 2, size.width - side / 2).toDouble();
    canvas.drawRect(
      Rect.fromLTWH(x - side / 2, 0, side, side),
      Paint()..color = signal.color == SignalColor.red ? dangerColor : greenColor,
    );
  }

  /// Bottom bar (the full-width foot zone, no outline). One filled yellow square only, at the left-right
  /// position of the braille path closest to the feet. When no braille block is on screen, the
  /// remembered direction stays (pinned to the left/right edge when outside the view) and fades out.
  void _drawBottomBar(Canvas canvas, Size size) {
    final m = marker;
    if (m == null || m.opacity <= 0) return;
    final bar = _px(GuideEngine.footZone, size);
    final side = bar.height;
    final x = (m.x * size.width).clamp(bar.left + side / 2, bar.right - side / 2).toDouble();
    canvas.drawRect(
      Rect.fromCenter(center: Offset(x, bar.center.dy), width: side, height: side),
      Paint()..color = markerColor.withValues(alpha: m.opacity),
    );
  }

  @override
  bool shouldRepaint(GuideOverlayPainter old) =>
      old.frame != frame ||
      old.blinkOn != blinkOn ||
      old.marker?.x != marker?.x ||
      old.marker?.opacity != marker?.opacity ||
      old.debugDetections != debugDetections;
}
