import 'dart:math' as math;
import 'dart:ui';

/// Turns the braille block pieces found by the detector into curved paths.
///
/// The model is trained to detect a braille strip as a stack of short pieces (one per horizontal
/// slice of the image). Pieces that continue each other are chained from the bottom of the screen
/// upward and drawn as one smooth band, so the overlay follows the strip like a road line.
/// All rectangles are normalized screen coordinates (0..1).
class BraillePath {
  /// Detected piece boxes run a little loose; draw slightly narrower than measured so the band stays on
  /// the blocks instead of spilling onto the pavement next to them.
  static const _widthScale = 0.9;
  static const _loneWidth = 0.6;

  /// Groups pieces into chains. Each chain starts with its lowest piece (closest to the feet).
  static List<List<Rect>> chains(List<Rect> pieces) {
    final sorted = [...pieces]..sort((a, b) => b.bottom.compareTo(a.bottom));
    final chains = <List<Rect>>[];
    for (final p in sorted) {
      List<Rect>? best;
      var bestScore = double.infinity;
      for (final c in chains) {
        final last = c.last;
        if (p.center.dy >= last.center.dy) continue; // must continue upward
        final gap = last.top - p.bottom; // > 0: gap between pieces, < 0: they overlap
        if (gap > math.max(last.height, p.height) * 0.8) continue;
        final dx = (p.center.dx - last.center.dx).abs();
        if (dx > math.max(last.width, p.width)) continue;
        final score = dx + gap.abs();
        if (score < bestScore) {
          bestScore = score;
          best = c;
        }
      }
      if (best != null) {
        best.add(p);
      } else {
        chains.add([p]);
      }
    }
    return chains;
  }

  /// Smooth band through a chain, in pixels.
  static Path band(List<Rect> chain, Size size) {
    Rect px(Rect r) => Rect.fromLTRB(r.left * size.width, r.top * size.height, r.right * size.width, r.bottom * size.height);
    if (chain.length == 1) {
      // A lone piece box is wider than the strip inside it (the strip runs across it at a slant).
      final r = px(chain.first);
      final narrow = Rect.fromCenter(center: r.center, width: r.width * _loneWidth, height: r.height);
      return Path()..addRRect(RRect.fromRectAndRadius(narrow, const Radius.circular(8)));
    }

    // Horizontal width of the strip inside each piece: a slanted strip makes its piece box wider than
    // the strip itself by about the sideways drift from one piece to the next.
    final widths = <double>[];
    for (var i = 0; i < chain.length; i++) {
      final prev = chain[math.max(i - 1, 0)].center.dx;
      final next = chain[math.min(i + 1, chain.length - 1)].center.dx;
      final span = math.min(i + 1, chain.length - 1) - math.max(i - 1, 0);
      final drift = span == 0 ? 0.0 : (next - prev).abs() / span;
      final w = chain[i].width;
      widths.add(math.max(w - drift, w * 0.4) * _widthScale * size.width);
    }

    // Control points: bottom edge of the first piece, every piece center, top edge of the last piece.
    final first = chain.first, last = chain.last;
    final pts = <Offset>[
      Offset(first.center.dx * size.width, first.bottom * size.height),
      for (final r in chain) Offset(r.center.dx * size.width, r.center.dy * size.height),
      Offset(last.center.dx * size.width, last.top * size.height),
    ];
    final ws = <double>[widths.first, ...widths, widths.last];

    // Catmull-Rom spline through the control points.
    const steps = 8;
    final center = <Offset>[];
    final halfW = <double>[];
    for (var i = 0; i < pts.length - 1; i++) {
      final p0 = pts[math.max(i - 1, 0)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[math.min(i + 2, pts.length - 1)];
      for (var k = 0; k < steps; k++) {
        final t = k / steps;
        center.add(_catmullRom(p0, p1, p2, p3, t));
        halfW.add((ws[i] + (ws[i + 1] - ws[i]) * t) / 2);
      }
    }
    center.add(pts.last);
    halfW.add(ws.last / 2);

    final path = Path()..moveTo(center.first.dx - halfW.first, center.first.dy);
    for (var i = 1; i < center.length; i++) {
      path.lineTo(center[i].dx - halfW[i], center[i].dy);
    }
    for (var i = center.length - 1; i >= 0; i--) {
      path.lineTo(center[i].dx + halfW[i], center[i].dy);
    }
    return path..close();
  }

  static Offset _catmullRom(Offset p0, Offset p1, Offset p2, Offset p3, double t) {
    final t2 = t * t, t3 = t2 * t;
    double f(double a, double b, double c, double d) =>
        0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
    return Offset(f(p0.dx, p1.dx, p2.dx, p3.dx), f(p0.dy, p1.dy, p2.dy, p3.dy));
  }
}
