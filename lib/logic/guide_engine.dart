import 'dart:math' as math;
import 'dart:ui';

import '../core/geometry.dart';
import '../core/settings.dart';
import '../core/types.dart';
import 'braille_smoother.dart';
import 'events.dart';
import 'street_rules.dart';

export 'events.dart';
export 'street_rules.dart' show StreetState, TrafficSignal, SignalColor;

/// A person/obstacle followed across frames, with smoothed distance and time-to-contact.
class TrackedObject {
  TrackedObject._(this.id, Detection d, DateTime now)
      : group = d.group,
        label = d.label,
        confidence = d.confidence,
        viewBox = d.viewBox,
        imageBox = d.imageBox,
        firstSeen = now,
        lastSeen = now,
        _heightEma = d.viewBox.height;

  final int id;
  final ObjectGroup group;
  String label;
  double confidence;
  Rect viewBox;
  Rect imageBox;
  DateTime firstSeen;
  DateTime lastSeen;
  int hits = 1;

  /// Smoothed ground distance in meters, null when the box bottom is above the horizon.
  double? distanceM;

  /// Share of the screen covered by the box (0..1).
  double get areaRatio => RectMath.area(viewBox).clamp(0.0, 1.0).toDouble();

  /// Seconds until contact estimated from how fast the box grows. Null when not approaching.
  double? ttcSec;

  bool inRoi = false;
  bool relevant = false; // in RoI and not too far: rendered and considered for warnings
  bool danger = false; // within the danger distance (hysteresis applied)
  bool onPath = false; // overlaps a braille block
  double pathIou = 0;

  double _heightEma;
  double? _growthEma;

  void _update(Detection d, DateTime now, double? rawDistance) {
    final dt = now.difference(lastSeen).inMicroseconds / 1e6;
    label = d.label;
    confidence = d.confidence;
    viewBox = RectMath.lerp(viewBox, d.viewBox, 0.6);
    imageBox = RectMath.lerp(imageBox, d.imageBox, 0.6);

    final h = d.viewBox.height;
    if (dt > 0.001 && dt < 1.0) {
      final growth = (h - _heightEma) / dt;
      _growthEma = _growthEma == null ? growth : _growthEma! * 0.7 + growth * 0.3;
    }
    _heightEma = _heightEma * 0.5 + h * 0.5;

    if (rawDistance == null) {
      distanceM = null;
    } else {
      distanceM = distanceM == null ? rawDistance : distanceM! * 0.6 + rawDistance * 0.4;
    }

    final g = _growthEma;
    ttcSec = (hits >= 5 && g != null && g > 0.02) ? _heightEma / g : null;

    hits++;
    lastSeen = now;
  }
}

/// Result of one engine update, consumed by the overlay and the audio service.
class GuideFrame {
  const GuideFrame({
    required this.brailleBoxes,
    this.damagedBrailleBoxes = const [],
    this.street = StreetState.empty,
    required this.objects,
    required this.events,
    required this.beepInterval,
    required this.obstacleOnPath,
    required this.brailleFollowing,
    required this.brailleAtFeet,
    required this.feetOccluded,
    required this.pitchDeg,
  });

  static const empty = GuideFrame(
    brailleBoxes: [],
    objects: [],
    events: [],
    beepInterval: null,
    obstacleOnPath: false,
    brailleFollowing: false,
    brailleAtFeet: false,
    feetOccluded: false,
    pitchDeg: 0,
  );

  final List<Rect> brailleBoxes; // normal braille pieces (yellow)
  final List<Rect> damagedBrailleBoxes; // damaged braille pieces (drawn in another colour)
  final StreetState street; // road / crosswalk / traffic light / caution zone
  final List<TrackedObject> objects; // only RoI objects within the max distance
  final List<GuideEvent> events;
  final Duration? beepInterval; // null = silent
  final bool obstacleOnPath;
  final bool brailleFollowing;
  final bool brailleAtFeet;
  final bool feetOccluded;
  final double pitchDeg;

  /// Every braille piece, damaged or not: both are guide paths for the rules and the direction marker.
  List<Rect> get allBrailleBoxes => [...brailleBoxes, ...damagedBrailleBoxes];
}

/// All warning logic, independent of Flutter widgets so it can be unit tested.
class GuideEngine {
  GuideEngine(this.settings);

  GuideSettings settings;

  /// 1. RoI: full height, horizontal 25%..75% of the screen.
  static const double roiLeft = 0.25;
  static const double roiRight = 0.75;

  /// Area right in front of the user's feet (bottom of the screen) for the "braille lost" check.
  static const Rect footZone = Rect.fromLTRB(0.0, 0.85, 1.0, 1.0);

  static const _trackTimeout = Duration(milliseconds: 350);
  static const _minHitsForDanger = 3;

  // Obstacle-on-path voice
  static const _overlapHold = Duration(milliseconds: 400);
  static const _overlapReset = Duration(seconds: 1);
  static const _obstacleCooldown = Duration(seconds: 5);
  static const _minIoa = 0.05;

  // Braille-lost voice
  static const _followAfter = Duration(milliseconds: 500);
  static const _footGapTolerance = Duration(milliseconds: 300);
  static const _lostAfter = Duration(milliseconds: 1500);
  static const _lostCooldown = Duration(seconds: 4);

  final List<TrackedObject> _tracks = [];
  final _brailleSmoother = BrailleSmoother();
  final _damagedSmoother = BrailleSmoother();
  final _roadSmoother = BrailleSmoother();
  final _crosswalkSmoother = BrailleSmoother();
  final _cautionSmoother = BrailleSmoother();
  final _street = StreetRules();
  int _nextId = 1;

  DateTime? _overlapSince;
  DateTime? _lastOverlap;
  bool _overlapAnnounced = false;
  DateTime? _lastObstacleVoice;

  bool _following = false;
  DateTime? _footSeenSince;
  DateTime? _lastFootSeen;
  DateTime? _lastLostVoice;

  GuideFrame update(List<Detection> detections, double rawPitchDeg, DateTime now) {
    final pitch = rawPitchDeg + settings.pitchOffsetDeg;
    // Floor pieces are smoothed over time (no shaking / one-frame flicker) before any rule uses them.
    List<Rect> boxesOf(ObjectGroup g) => [for (final d in detections) if (d.group == g) d.viewBox];
    final brailleNormal = _brailleSmoother.update(boxesOf(ObjectGroup.braille), now);
    final brailleDamaged = _damagedSmoother.update(boxesOf(ObjectGroup.brailleDamaged), now);
    final braille = [...brailleNormal, ...brailleDamaged]; // both guide the user
    final road = _roadSmoother.update(boxesOf(ObjectGroup.road), now);
    final crosswalk = _crosswalkSmoother.update(boxesOf(ObjectGroup.crosswalk), now);
    final caution = _cautionSmoother.update(boxesOf(ObjectGroup.caution), now);
    final signals = [
      for (final d in detections)
        if (d.group == ObjectGroup.trafficRed)
          TrafficSignal(SignalColor.red, d.viewBox)
        else if (d.group == ObjectGroup.trafficGreen)
          TrafficSignal(SignalColor.green, d.viewBox),
    ];
    final others = detections
        .where((d) => d.group == ObjectGroup.person || d.group == ObjectGroup.obstacle)
        .toList();

    _track(others, pitch, now);

    // --- 1 + 2: RoI filtering, distance, danger ---------------------------------------------
    final relevant = <TrackedObject>[];
    for (final t in _tracks) {
      // In the path when the object's foot point is inside the band, or when it blocks at least half of
      // the band. A wide box at the side that only grazes the band (desks, parked cars) is ignored.
      final footX = t.viewBox.center.dx;
      final overlap = math.min(t.viewBox.right, roiRight) - math.max(t.viewBox.left, roiLeft);
      t.inRoi = (footX >= roiLeft && footX <= roiRight) || overlap >= (roiRight - roiLeft) * 0.5;
      final d = t.distanceM;
      final bigEnough = t.areaRatio >= settings.dangerAreaRatio;
      final tooFar = (d == null || d > settings.maxDistanceM) && !bigEnough;
      t.relevant = t.inRoi && !tooFar;

      if (!t.relevant) {
        t.danger = false;
      } else if (!t.danger) {
        final near = d != null && d <= settings.dangerDistanceM;
        t.danger = t.hits >= _minHitsForDanger && (near || bigEnough);
      } else {
        final stillNear = d != null && d <= settings.dangerDistanceM + 0.4;
        final stillBig = t.areaRatio >= settings.dangerAreaRatio * 0.8;
        t.danger = stillNear || stillBig;
      }
      if (t.relevant) relevant.add(t);
    }

    // --- 4a: dangerous object overlapping a braille block -----------------------------------
    var anyOnPath = false;
    for (final t in relevant) {
      t.onPath = false;
      t.pathIou = 0;
      if (!t.danger) continue;
      final foot = Offset(t.viewBox.center.dx, t.viewBox.bottom);
      for (final b in braille) {
        final overlaps = RectMath.ioa(t.viewBox, b) >= _minIoa || b.contains(foot);
        if (overlaps) {
          t.onPath = true;
          t.pathIou = math.max(t.pathIou, RectMath.iou(t.viewBox, b));
        }
      }
      anyOnPath = anyOnPath || t.onPath;
    }

    final events = <GuideEvent>[];
    if (anyOnPath) {
      _lastOverlap = now;
      _overlapSince ??= now;
      final held = now.difference(_overlapSince!) >= _overlapHold;
      final cooled = _lastObstacleVoice == null || now.difference(_lastObstacleVoice!) >= _obstacleCooldown;
      if (!_overlapAnnounced && held && cooled) {
        events.add(GuideEvent.obstacleOnPath);
        _overlapAnnounced = true;
        _lastObstacleVoice = now;
      }
    } else if (_lastOverlap == null || now.difference(_lastOverlap!) > _overlapReset) {
      _overlapSince = null;
      _overlapAnnounced = false;
    }

    // --- 4b: braille block disappeared from under the user's feet ---------------------------
    final footArea = RectMath.area(footZone);
    // Pieces of a narrow strip are small, so a little overlap with the (full-width) foot zone is enough.
    final brailleAtFeet = braille.any((b) => RectMath.intersection(b, footZone) >= footArea * 0.02);
    final feetOccluded = _tracks.any(
      (t) => now.difference(t.lastSeen) <= _trackTimeout && RectMath.intersection(t.viewBox, footZone) >= footArea * 0.3,
    );

    if (brailleAtFeet) {
      final gap = _lastFootSeen == null ? null : now.difference(_lastFootSeen!);
      if (_footSeenSince == null || (gap != null && gap > _footGapTolerance)) {
        _footSeenSince = now;
      }
      _lastFootSeen = now;
      if (!_following && now.difference(_footSeenSince!) >= _followAfter) {
        _following = true;
      }
    } else if (_following) {
      if (feetOccluded) {
        _lastFootSeen = now; // something stands on the blocks: hidden is not the same as gone
      } else if (_lastFootSeen != null && now.difference(_lastFootSeen!) >= _lostAfter) {
        _following = false;
        _footSeenSince = null;
        final cooled = _lastLostVoice == null || now.difference(_lastLostVoice!) >= _lostCooldown;
        if (cooled) {
          events.add(GuideEvent.brailleLost);
          _lastLostVoice = now;
        }
      }
    } else if (_lastFootSeen != null && now.difference(_lastFootSeen!) > _footGapTolerance) {
      _footSeenSince = null;
    }

    // --- Road entry, crosswalk, traffic light, caution zone ---------------------------------
    final street = _street.update(
      road: road,
      crosswalk: crosswalk,
      caution: caution,
      signals: signals,
      now: now,
      events: events,
    );

    // --- 2: beep interval from the closest danger -------------------------------------------
    Duration? beep;
    final dangers = relevant.where((t) => t.danger);
    if (dangers.isNotEmpty) {
      var p = 0.0;
      var urgent = false;
      for (final t in dangers) {
        p = math.max(p, _proximity(t));
        final ttc = t.ttcSec;
        if (ttc != null && ttc < 1.5) urgent = true;
      }
      final ms = urgent ? 150 : (1000 - 850 * p).round().clamp(150, 1000).toInt();
      beep = Duration(milliseconds: ms);
    }

    relevant.sort((a, b) => (a.distanceM ?? 99).compareTo(b.distanceM ?? 99));
    return GuideFrame(
      brailleBoxes: brailleNormal,
      damagedBrailleBoxes: brailleDamaged,
      street: street,
      objects: relevant,
      events: events,
      beepInterval: beep,
      obstacleOnPath: anyOnPath,
      brailleFollowing: _following,
      brailleAtFeet: brailleAtFeet,
      feetOccluded: feetOccluded,
      pitchDeg: pitch,
    );
  }

  /// 0 at the edge of the danger zone, 1 when very close.
  double _proximity(TrackedObject t) {
    var p = 0.0;
    final d = t.distanceM;
    final dz = settings.dangerDistanceM;
    if (d != null && dz > 0.5) p = ((dz - d) / (dz - 0.5)).clamp(0.0, 1.0).toDouble();
    final a0 = settings.dangerAreaRatio;
    if (a0 < 0.5) {
      final pa = ((t.areaRatio - a0) / (0.5 - a0)).clamp(0.0, 1.0).toDouble();
      p = math.max(p, pa);
    }
    return p;
  }

  void _track(List<Detection> dets, double pitch, DateTime now) {
    // Greedy IoU matching within the same group.
    final pairs = <(double, int, int)>[];
    for (var ti = 0; ti < _tracks.length; ti++) {
      for (var di = 0; di < dets.length; di++) {
        if (_tracks[ti].group != dets[di].group) continue;
        final iou = RectMath.iou(_tracks[ti].viewBox, dets[di].viewBox);
        if (iou >= 0.25) pairs.add((iou, ti, di));
      }
    }
    pairs.sort((a, b) => b.$1.compareTo(a.$1));
    final usedT = <int>{};
    final usedD = <int>{};
    for (final (_, ti, di) in pairs) {
      if (usedT.contains(ti) || usedD.contains(di)) continue;
      usedT.add(ti);
      usedD.add(di);
      _tracks[ti]._update(dets[di], now, _distance(dets[di].imageBox, pitch));
    }
    for (var di = 0; di < dets.length; di++) {
      if (usedD.contains(di)) continue;
      final t = TrackedObject._(_nextId++, dets[di], now);
      t.distanceM = _distance(dets[di].imageBox, pitch);
      _tracks.add(t);
    }
    _tracks.removeWhere((t) => now.difference(t.lastSeen) > _trackTimeout);
  }

  double? _distance(Rect imageBox, double pitch) => GroundDistance.estimate(
        yNorm: imageBox.bottom.clamp(0.0, 1.0).toDouble(),
        pitchDeg: pitch,
        cameraHeightM: settings.cameraHeightM,
        verticalFovDeg: settings.verticalFovDeg,
      );

  void reset() {
    _tracks.clear();
    _brailleSmoother.reset();
    _damagedSmoother.reset();
    _roadSmoother.reset();
    _crosswalkSmoother.reset();
    _cautionSmoother.reset();
    _street.reset();
    _overlapSince = null;
    _lastOverlap = null;
    _overlapAnnounced = false;
    _following = false;
    _footSeenSince = null;
    _lastFootSeen = null;
  }
}
