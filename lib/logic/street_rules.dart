import 'dart:ui';

import '../core/geometry.dart';
import 'events.dart';

enum SignalColor { red, green }

class TrafficSignal {
  const TrafficSignal(this.color, this.box);

  final SignalColor color;
  final Rect box; // normalized screen coordinates
}

/// What the street rules want drawn this frame.
class StreetState {
  const StreetState({
    this.roadWarning = false,
    this.roadBoxes = const [],
    this.cautionBoxes = const [],
    this.signal,
    this.crosswalkContext = false,
  });

  static const empty = StreetState();

  final bool roadWarning; // walking onto the road: road pieces are drawn red
  final List<Rect> roadBoxes;
  final List<Rect> cautionBoxes; // caution-zone pieces in the walking path
  final TrafficSignal? signal; // shown only near a crosswalk
  final bool crosswalkContext; // a crosswalk was seen in the last [StreetRules.crosswalkMemory]
}

/// Road entry, crosswalk, traffic light and caution-zone rules. All boxes are normalized screen
/// coordinates and already smoothed over time.
class StreetRules {
  /// Path ahead of the feet: road covering a good part of it means stepping onto the road.
  static const roadZone = Rect.fromLTRB(0.25, 0.6, 0.75, 1.0);

  /// Lower part of the view: a crosswalk here means the user is arriving at it.
  static const crosswalkZone = Rect.fromLTRB(0.1, 0.4, 0.9, 1.0);

  /// Walking path near the feet for caution zones (manholes, gratings, stairs...).
  static const cautionZone = Rect.fromLTRB(0.25, 0.6, 0.75, 1.0);

  static const crosswalkMemory = Duration(seconds: 20);

  static const _roadOn = 0.25, _roadOff = 0.08;
  static const _roadHold = Duration(milliseconds: 500);
  static const _roadRearm = Duration(seconds: 2);
  static const _roadCooldown = Duration(seconds: 8);

  static const _crosswalkOn = 0.03;
  static const _crosswalkHold = Duration(milliseconds: 500);
  static const _crosswalkRearm = Duration(seconds: 5);
  static const _crosswalkCooldown = Duration(seconds: 20);

  static const _signalFrames = 3;
  static const _signalMissing = Duration(seconds: 1);
  static const _signalForget = Duration(seconds: 10); // after this long without a light, announce again

  static const _cautionOn = 0.05;
  static const _cautionHold = Duration(milliseconds: 400);
  static const _cautionRearm = Duration(seconds: 2);
  static const _cautionCooldown = Duration(seconds: 6);

  DateTime? _roadSince, _roadLast, _roadVoice;
  bool _roadArmed = true;

  DateTime? _cwSince, _cwLast, _cwVoice, _cwConfirmed;
  bool _cwArmed = true;

  SignalColor? _candColor, _announcedColor;
  int _candCount = 0;
  Rect? _candBox;
  DateTime? _signalLast;

  DateTime? _cautionSince, _cautionLast, _cautionVoice;
  bool _cautionArmed = true;

  void reset() {
    _roadSince = _roadLast = null;
    _roadArmed = true;
    _cwSince = _cwLast = _cwConfirmed = null;
    _cwArmed = true;
    _candColor = _announcedColor = null;
    _candCount = 0;
    _candBox = null;
    _signalLast = null;
    _cautionSince = _cautionLast = null;
    _cautionArmed = true;
  }

  static double coverage(List<Rect> pieces, Rect zone) {
    var sum = 0.0;
    for (final p in pieces) {
      sum += RectMath.intersection(p, zone);
    }
    final area = RectMath.area(zone);
    return area <= 0 ? 0.0 : (sum / area).clamp(0.0, 1.0).toDouble();
  }

  static bool _cooled(DateTime? last, DateTime now, Duration cooldown) =>
      last == null || now.difference(last) >= cooldown;

  StreetState update({
    required List<Rect> road,
    required List<Rect> crosswalk,
    required List<Rect> caution,
    required List<TrafficSignal> signals,
    required DateTime now,
    required List<GuideEvent> events,
  }) {
    // --- Crosswalk ahead -------------------------------------------------------------------
    if (coverage(crosswalk, crosswalkZone) >= _crosswalkOn) {
      _cwSince ??= now;
      _cwLast = now;
    } else if (_cwLast == null || now.difference(_cwLast!) > _crosswalkRearm) {
      _cwSince = null;
      _cwArmed = true;
    }
    final cwHeld = _cwSince != null && now.difference(_cwSince!) >= _crosswalkHold;
    if (cwHeld) _cwConfirmed = now;
    if (cwHeld && _cwArmed && _cooled(_cwVoice, now, _crosswalkCooldown)) {
      events.add(GuideEvent.crosswalk);
      _cwArmed = false;
      _cwVoice = now;
    }
    // Traffic light voice and road-warning suppression apply after the crosswalk is confirmed,
    // so "횡단보도입니다" always comes before the signal colour.
    final crosswalkContext = _cwConfirmed != null && now.difference(_cwConfirmed!) <= crosswalkMemory;

    // --- Stepping onto the road (not while at a crosswalk) -----------------------------------
    final roadCov = coverage(road, roadZone);
    if (roadCov >= _roadOn) {
      _roadSince ??= now;
      _roadLast = now;
    } else if (roadCov >= _roadOff) {
      if (_roadSince != null) _roadLast = now; // in between: keep the current state
    } else if (_roadLast == null || now.difference(_roadLast!) > _roadRearm) {
      _roadSince = null;
      _roadArmed = true;
    }
    final roadActive = _roadSince != null && now.difference(_roadSince!) >= _roadHold && !crosswalkContext;
    if (roadActive && _roadArmed && _cooled(_roadVoice, now, _roadCooldown)) {
      events.add(GuideEvent.roadAhead);
      _roadArmed = false;
      _roadVoice = now;
    }

    // --- Pedestrian traffic light colour ----------------------------------------------------
    // The model only learned pedestrian signals (vertical ones), so no crosswalk is required.
    TrafficSignal? best;
    for (final s in signals) {
      final cx = s.box.center.dx;
      if (cx < 0.15 || cx > 0.85) continue;
      if (best == null || RectMath.area(s.box) > RectMath.area(best.box)) best = s;
    }
    if (best != null) {
      _candCount = best.color == _candColor ? _candCount + 1 : 1;
      _candColor = best.color;
      _candBox = best.box;
      _signalLast = now;
    } else if (_signalLast == null || now.difference(_signalLast!) > _signalMissing) {
      _candColor = null;
      _candCount = 0;
      _candBox = null;
    }
    final stable = _candCount >= _signalFrames ? _candColor : null;
    TrafficSignal? shown;
    if (_signalLast == null || now.difference(_signalLast!) > _signalForget) {
      _announcedColor = null;
    }
    if (stable != null) {
      shown = TrafficSignal(stable, _candBox!);
      if (stable != _announcedColor) {
        events.add(stable == SignalColor.red ? GuideEvent.signalRed : GuideEvent.signalGreen);
        _announcedColor = stable;
      }
    }

    // --- Caution zone in the walking path --------------------------------------------------
    final inPath = [for (final c in caution) if (RectMath.intersection(c, cautionZone) > 0) c];
    if (coverage(caution, cautionZone) >= _cautionOn) {
      _cautionSince ??= now;
      _cautionLast = now;
    } else if (_cautionLast == null || now.difference(_cautionLast!) > _cautionRearm) {
      _cautionSince = null;
      _cautionArmed = true;
    }
    if (_cautionSince != null &&
        now.difference(_cautionSince!) >= _cautionHold &&
        _cautionArmed &&
        _cooled(_cautionVoice, now, _cautionCooldown)) {
      events.add(GuideEvent.caution);
      _cautionArmed = false;
      _cautionVoice = now;
    }

    return StreetState(
      roadWarning: roadActive,
      roadBoxes: roadActive ? [for (final r in road) if (r.bottom > 0.5) r] : const [],
      cautionBoxes: inPath,
      signal: shown,
      crosswalkContext: crosswalkContext,
    );
  }
}
