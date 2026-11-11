// Unit tests for the warning logic (no camera needed): flutter test
import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:walk_guide/core/geometry.dart';
import 'package:walk_guide/core/settings.dart';
import 'package:walk_guide/core/types.dart';
import 'package:walk_guide/logic/braille_marker.dart';
import 'package:walk_guide/logic/braille_path.dart';
import 'package:walk_guide/logic/braille_smoother.dart';
import 'package:walk_guide/logic/guide_engine.dart';

Detection det(ObjectGroup g, Rect r) =>
    Detection(group: g, label: g.name, confidence: 0.9, imageBox: r, viewBox: r);

/// A braille strip as the model reports it: a stack of short slices.
List<Detection> strip(double left, double top, double right, double bottom) => [
      for (var y = top; y < bottom - 1e-9; y += 0.1)
        det(ObjectGroup.braille, Rect.fromLTRB(left, y, right, (y + 0.1).clamp(0.0, bottom).toDouble())),
    ];

void main() {
  final t0 = DateTime(2026, 1, 1);
  DateTime at(int ms) => t0.add(Duration(milliseconds: ms));

  test('ground distance: lower in the image means closer', () {
    final far = GroundDistance.estimate(yNorm: 0.6, pitchDeg: 20, cameraHeightM: 1.3, verticalFovDeg: 67)!;
    final near = GroundDistance.estimate(yNorm: 0.95, pitchDeg: 20, cameraHeightM: 1.3, verticalFovDeg: 67)!;
    expect(near, lessThan(far));
    expect(GroundDistance.estimate(yNorm: 0.2, pitchDeg: 0, cameraHeightM: 1.3, verticalFovDeg: 67), isNull);
  });

  test('view mapping crops top/bottom of a 4:3 frame on a wide landscape screen', () {
    final r = ViewMapper.imageToView(const Rect.fromLTRB(0, 0.5, 1, 0.5), 4 / 3, const Size(900, 400));
    expect(r.top, closeTo(0.5, 1e-9)); // center stays center
    expect(r.left, closeTo(0, 1e-9));
    expect(r.right, closeTo(1, 1e-9));
  });

  test('pitch is the same in portrait and landscape', () {
    final portrait = GroundDistance.pitchFromAccelerometer(0, 8.5, 4.9);
    final landscape = GroundDistance.pitchFromAccelerometer(8.5, 0, 4.9);
    expect(portrait, closeTo(landscape, 1e-9));
    expect(portrait, closeTo(30, 0.5));
  });

  test('a wide box that only grazes the band is ignored', () {
    final engine = GuideEngine(GuideSettings());
    final f = engine.update([det(ObjectGroup.obstacle, const Rect.fromLTRB(0.0, 0.3, 0.47, 1.0))], 25, at(0));
    expect(f.objects, isEmpty);
  });

  test('objects outside the 25-75% band are ignored', () {
    final engine = GuideEngine(GuideSettings());
    final f = engine.update([det(ObjectGroup.person, const Rect.fromLTRB(0.0, 0.3, 0.2, 0.98))], 25, at(0));
    expect(f.objects, isEmpty);
    expect(f.beepInterval, isNull);
  });

  test('a close object in the RoI becomes dangerous and beeps', () {
    final engine = GuideEngine(GuideSettings());
    late GuideFrame f;
    for (var i = 0; i < 4; i++) {
      f = engine.update([det(ObjectGroup.obstacle, const Rect.fromLTRB(0.4, 0.5, 0.6, 0.97))], 25, at(i * 100));
    }
    expect(f.objects.single.danger, isTrue);
    expect(f.beepInterval, isNotNull);
  });

  test('obstacle on a braille block announces once', () {
    final engine = GuideEngine(GuideSettings());
    final braille = strip(0.35, 0.4, 0.65, 1.0);
    final obstacle = det(ObjectGroup.obstacle, const Rect.fromLTRB(0.42, 0.6, 0.58, 0.97));
    final events = <GuideEvent>[];
    for (var i = 0; i < 20; i++) {
      events.addAll(engine.update([...braille, obstacle], 25, at(i * 100)).events);
    }
    expect(events.where((e) => e == GuideEvent.obstacleOnPath).length, 1);
  });

  test('braille lost fires after following and then losing the blocks', () {
    final engine = GuideEngine(GuideSettings());
    final braille = strip(0.4, 0.5, 0.6, 1.0);
    for (var i = 0; i < 10; i++) {
      engine.update(braille, 25, at(i * 100));
    }
    final events = <GuideEvent>[];
    for (var i = 10; i < 40; i++) {
      events.addAll(engine.update([], 25, at(i * 100)).events);
    }
    expect(events, contains(GuideEvent.brailleLost));
  });

  test('no braille-lost while something covers the feet', () {
    final engine = GuideEngine(GuideSettings());
    final braille = strip(0.4, 0.5, 0.6, 1.0);
    final person = det(ObjectGroup.person, const Rect.fromLTRB(0.3, 0.4, 0.7, 1.0));
    for (var i = 0; i < 10; i++) {
      engine.update(braille, 25, at(i * 100));
    }
    final events = <GuideEvent>[];
    for (var i = 10; i < 40; i++) {
      events.addAll(engine.update([person], 25, at(i * 100)).events);
    }
    expect(events, isNot(contains(GuideEvent.brailleLost)));
  });

  test('braille pieces: one slanted strip becomes one chain, a second strip stays separate', () {
    // A strip going up and to the left, sliced into 4 pieces, plus an unrelated piece far to the right.
    final strip = [
      for (var i = 0; i < 4; i++)
        Rect.fromLTWH(0.50 - i * 0.03, 0.90 - i * 0.10, 0.10, 0.10),
    ];
    const other = Rect.fromLTWH(0.90, 0.85, 0.08, 0.10);
    final chains = BraillePath.chains([...strip.reversed, other]);
    expect(chains.length, 2);
    final main = chains.firstWhere((c) => c.length == 4);
    expect(main.first, strip.first); // starts at the piece closest to the feet
    expect(main.last, strip.last);
    final path = BraillePath.band(main, const Size(800, 400));
    expect(path.getBounds().width, greaterThan(0));
  });

  test('braille marker: stays at the edge after turning away, then fades out', () {
    final m = BrailleMarker();
    const fov = 66.0;
    // Block seen slightly right of center while heading 0.
    expect(m.update(visibleX: 0.6, headingDeg: 0, hfovDeg: fov, now: at(0))!.remembered, isFalse);
    // Turn 90 degrees left: the block is now far to the right -> pinned to the right edge.
    final s1 = m.update(visibleX: null, headingDeg: -90, hfovDeg: fov, now: at(1000))!;
    expect(s1.remembered, isTrue);
    expect(s1.x, 1.0);
    expect(s1.opacity, 1.0);
    // Turn right past it: now it is on the left.
    expect(m.update(visibleX: null, headingDeg: 90, hfovDeg: fov, now: at(2000))!.x, 0.0);
    // Fading after 5 s, gone after 8 s.
    expect(m.update(visibleX: null, headingDeg: 90, hfovDeg: fov, now: at(6500))!.opacity, lessThan(1.0));
    expect(m.update(visibleX: null, headingDeg: 90, hfovDeg: fov, now: at(8100)), isNull);
  });

  test('braille marker: looking back where the block was and not finding it removes the marker', () {
    final m = BrailleMarker();
    const fov = 66.0;
    m.update(visibleX: 0.5, headingDeg: 0, hfovDeg: fov, now: at(0));
    expect(m.update(visibleX: null, headingDeg: 0, hfovDeg: fov, now: at(500)), isNotNull); // short gap kept
    expect(m.update(visibleX: null, headingDeg: 0, hfovDeg: fov, now: at(1600)), isNull); // block ended
  });

  test('braille smoother: one-frame hit hidden, small jumps softened, short gaps bridged', () {
    final sm = BrailleSmoother();
    const a = Rect.fromLTWH(0.40, 0.80, 0.10, 0.10);
    const b = Rect.fromLTWH(0.42, 0.80, 0.10, 0.10);
    expect(sm.update([a], at(0)), isEmpty); // seen once: not shown yet
    final shown = sm.update([b], at(100)).single; // moved halfway toward b
    expect(shown.left, closeTo(0.41, 1e-9));
    expect(sm.update([], at(250)).length, 1); // missed one frame: kept
    expect(sm.update([], at(400)), isEmpty); // gone after 250 ms without a detection
  });

  test('road: stepping onto the road warns once, not again while staying on it', () {
    final engine = GuideEngine(GuideSettings());
    final road = [for (var i = 0; i < 3; i++) Rect.fromLTRB(0.2, 0.7 + i * 0.1, 0.8, 0.8 + i * 0.1)];
    final events = <GuideEvent>[];
    late GuideFrame f;
    for (var i = 0; i < 40; i++) {
      f = engine.update([for (final r in road) det(ObjectGroup.road, r)], 25, at(i * 100));
      events.addAll(f.events);
    }
    expect(events.where((e) => e == GuideEvent.roadAhead).length, 1);
    expect(f.street.roadWarning, isTrue);
  });

  test('crosswalk: announced, then the traffic light colour and its change; no road warning there', () {
    final engine = GuideEngine(GuideSettings());
    const cw = Rect.fromLTRB(0.2, 0.6, 0.8, 0.75);
    final road = [for (var i = 0; i < 3; i++) Rect.fromLTRB(0.2, 0.7 + i * 0.1, 0.8, 0.8 + i * 0.1)];
    const light = Rect.fromLTRB(0.48, 0.1, 0.52, 0.2);
    final events = <GuideEvent>[];
    late GuideFrame f;
    for (var i = 0; i < 30; i++) {
      final group = i < 20 ? ObjectGroup.trafficRed : ObjectGroup.trafficGreen;
      f = engine.update([
        det(ObjectGroup.crosswalk, cw),
        for (final r in road) det(ObjectGroup.road, r),
        det(group, light),
      ], 25, at(i * 100));
      events.addAll(f.events);
    }
    expect(events.where((e) => e == GuideEvent.crosswalk).length, 1);
    expect(events.where((e) => e == GuideEvent.signalRed).length, 1);
    expect(events.where((e) => e == GuideEvent.signalGreen).length, 1);
    expect(events, isNot(contains(GuideEvent.roadAhead)));
    expect(f.street.signal?.color, SignalColor.green);
  });

  test('pedestrian light alone is announced once, not repeated while it stays red', () {
    final engine = GuideEngine(GuideSettings());
    final events = <GuideEvent>[];
    for (var i = 0; i < 20; i++) {
      events.addAll(engine.update([det(ObjectGroup.trafficRed, const Rect.fromLTRB(0.48, 0.1, 0.52, 0.2))], 25,
          at(i * 100)).events);
    }
    expect(events.where((e) => e == GuideEvent.signalRed).length, 1);
  });

  test('caution zone in the path warns once per encounter', () {
    final engine = GuideEngine(GuideSettings());
    const manhole = Rect.fromLTRB(0.4, 0.75, 0.6, 0.9);
    final events = <GuideEvent>[];
    for (var i = 0; i < 20; i++) {
      events.addAll(engine.update([det(ObjectGroup.caution, manhole)], 25, at(i * 100)).events);
    }
    expect(events.where((e) => e == GuideEvent.caution).length, 1);
  });

  test('bollard and damaged braille map to obstacle and braille paths', () {
    expect(ClassMapper.map('bollard'), ObjectGroup.obstacle);
    expect(ClassMapper.map('braille_damaged'), ObjectGroup.brailleDamaged);
    expect(ClassMapper.map('traffic_light_green'), ObjectGroup.trafficGreen);
  });
}
