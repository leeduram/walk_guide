import 'dart:ui';

import '../core/geometry.dart';

/// Steadies braille block pieces over time so the yellow band does not shake or flicker.
///
/// Each detected piece is matched to the piece from the previous frames it overlaps most; its box
/// moves halfway toward the new detection instead of jumping. A piece missed for a frame or two is kept
/// for [keepAlive], and a piece seen in only one frame is not shown (removes one-frame false hits).
class BrailleSmoother {
  static const keepAlive = Duration(milliseconds: 250);
  static const _newWeight = 0.5;
  static const _minIou = 0.2;

  final List<_Piece> _pieces = [];

  List<Rect> update(List<Rect> detections, DateTime now) {
    final used = <_Piece>{};
    for (final d in detections) {
      _Piece? best;
      var bestIou = _minIou;
      for (final p in _pieces) {
        if (used.contains(p)) continue;
        final iou = RectMath.iou(p.box, d);
        if (iou > bestIou) {
          bestIou = iou;
          best = p;
        }
      }
      if (best != null) {
        best.box = RectMath.lerp(best.box, d, _newWeight);
        best.lastSeen = now;
        best.hits++;
        used.add(best);
      } else {
        final p = _Piece(d, now);
        _pieces.add(p);
        used.add(p);
      }
    }
    _pieces.removeWhere((p) => now.difference(p.lastSeen) > keepAlive);
    return [for (final p in _pieces) if (p.hits >= 2) p.box];
  }

  void reset() => _pieces.clear();
}

class _Piece {
  _Piece(this.box, this.lastSeen);

  Rect box;
  DateTime lastSeen;
  int hits = 1;
}
