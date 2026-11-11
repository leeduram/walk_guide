import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../audio/audio_service.dart';
import '../core/geometry.dart';
import '../core/settings.dart';
import '../core/types.dart';
import '../logic/braille_marker.dart';
import '../logic/braille_path.dart';
import '../logic/guide_engine.dart';
import 'overlay_painter.dart';

/// Trained model produced by training/export_tflite.py. When it is missing the app falls back to the
/// bundled official COCO model (people and common obstacles only, no braille blocks) so it can be tested.
/// Both ship inside the APK, so no download is needed after a reinstall.
/// TEMPORARY test mode: draws every raw detection as a thin labeled box plus the road / crosswalk
/// zones, to see what the model finds. Set to false (or delete) before the final build.
const bool kShowDetections = false;

const String customModelAsset = 'assets/models/walk_guide.tflite';
const String fallbackModelAsset = 'assets/models/yolo26n_coco.tflite';
const String fallbackModelId = 'yolo26n'; // downloaded on first run only if the bundled file is missing

class GuideScreen extends StatefulWidget {
  const GuideScreen({super.key});

  @override
  State<GuideScreen> createState() => _GuideScreenState();
}

class _GuideScreenState extends State<GuideScreen> with WidgetsBindingObserver {
  final _controller = YOLOViewController();
  final _audio = AudioService();

  GuideSettings? _settings;
  GuideEngine? _engine;
  Widget? _yoloView;
  String? _error;

  GuideFrame _frame = GuideFrame.empty;
  List<Detection> _rawDetections = const [];
  Size _viewSize = Size.zero;
  double _imageAspect = 4 / 3; // 4:3 camera frame in landscape, refined from the first detections
  double _pitch = 0;
  // Low-passed gravity (accelerometer) and the camera heading integrated from the gyroscope.
  double _ax = 0, _ay = 0, _az = 9.8;
  double _headingDeg = 0;
  DateTime? _lastGyro;
  final _marker = BrailleMarker();
  BrailleMarkerState? _markerState;
  bool _blinkOn = false;

  StreamSubscription<AccelerometerEvent>? _accel;
  StreamSubscription<GyroscopeEvent>? _gyro;
  Timer? _blinkTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WakelockPlus.enable();
    _controller.setShowOverlays(false); // we draw our own overlay
    // The plugin keeps only the 30 most confident detections by default. Floor classes come as many
    // pieces (braille + road + crosswalk can exceed 30), so allow the maximum.
    unawaited(_controller.setNumItemsThreshold(100).catchError((_) {}));
    _init();
  }

  Future<void> _init() async {
    // No settings screen: the app is for low-vision users, so it always runs with the built-in values.
    final settings = GuideSettings();
    final engine = GuideEngine(settings);
    try {
      await _audio.init();
    } catch (e) {
      debugPrint('audio init failed: $e');
    }
    _applyAudioSettings(settings);

    final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final assets = manifest.listAssets();
    final hasCustom = assets.contains(customModelAsset);
    final modelPath = hasCustom
        ? customModelAsset
        : (assets.contains(fallbackModelAsset) ? fallbackModelAsset : fallbackModelId);

    _accel = accelerometerEventStream(samplingPeriod: SensorInterval.gameInterval).listen((e) {
      final p = GroundDistance.pitchFromAccelerometer(e.x, e.y, e.z);
      _pitch = _pitch * 0.9 + p * 0.1; // low-pass: hand shake and footsteps
      _ax = _ax * 0.9 + e.x * 0.1;
      _ay = _ay * 0.9 + e.y * 0.1;
      _az = _az * 0.9 + e.z * 0.1;
    });

    // Turning left/right = rotation around gravity. Works in any phone orientation.
    _gyro = gyroscopeEventStream(samplingPeriod: SensorInterval.gameInterval).listen((e) {
      final now = DateTime.now();
      final last = _lastGyro;
      _lastGyro = now;
      if (last == null) return;
      final dt = now.difference(last).inMicroseconds / 1e6;
      if (dt <= 0 || dt > 0.2) return;
      final g = math.sqrt(_ax * _ax + _ay * _ay + _az * _az);
      if (g < 1) return;
      final yawLeft = (e.x * _ax + e.y * _ay + e.z * _az) / g; // rad/s, counter-clockwise seen from above
      _headingDeg -= yawLeft * dt * 180 / math.pi;
    });

    _blinkTimer = Timer.periodic(const Duration(milliseconds: 250), (_) {
      final anyDanger = _frame.objects.any((t) => t.danger);
      if (!mounted) return;
      if (anyDanger || _blinkOn) {
        setState(() => _blinkOn = anyDanger && !_blinkOn);
      }
    });

    if (!mounted) return;
    setState(() {
      _settings = settings;
      _engine = engine;
      _yoloView = YOLOView(
        key: const ValueKey('yolo'),
        modelPath: modelPath,
        task: YOLOTask.detect,
        controller: _controller,
        confidenceThreshold: [settings.confidence, settings.brailleConfidence, settings.surfaceConfidence]
            .reduce(math.min),
        iouThreshold: 0.5,
        useGpu: true,
        onResult: _onResult,
        onModelError: (error, path, task) {
          if (mounted) setState(() => _error = '모델을 불러오지 못했습니다: $error');
        },
      );
    });
  }

  void _applyAudioSettings(GuideSettings s) {
    _audio.beepEnabled = s.beepEnabled;
    _audio.voiceEnabled = s.voiceEnabled;
  }

  // ------------------------------------------------------------------------------------------
  // Per-frame pipeline: detections -> engine -> overlay + audio
  // ------------------------------------------------------------------------------------------
  void _onResult(List<YOLOResult> results) {
    final engine = _engine;
    if (engine == null || !mounted) return;
    final detections = _convert(results);
    _rawDetections = detections;
    final now = DateTime.now();
    final frame = engine.update(detections, _pitch, now);
    // Bottom-bar marker: the braille path closest to the feet, remembered after it leaves the screen.
    final chains = BraillePath.chains(frame.allBrailleBoxes);
    double? visibleX;
    if (chains.isNotEmpty) {
      final nearest = chains.reduce((a, b) => a.first.bottom >= b.first.bottom ? a : b);
      visibleX = nearest.first.center.dx;
    }
    _markerState = _marker.update(
      visibleX: visibleX,
      headingDeg: _headingDeg,
      hfovDeg: _horizontalFovDeg(_settings),
      now: now,
    );
    _audio.beepInterval = frame.beepInterval;
    for (final event in frame.events) {
      _audio.speak(event);
      HapticFeedback.heavyImpact();
    }
    setState(() => _frame = frame);
  }

  List<Detection> _convert(List<YOLOResult> results) {
    final out = <Detection>[];
    for (final r in results) {
      final group = ClassMapper.map(r.className);
      if (group == null) continue;
      final s = _settings;
      if (s != null) {
        final minConf = switch (group) {
          ObjectGroup.braille || ObjectGroup.brailleDamaged => s.brailleConfidence,
          ObjectGroup.road || ObjectGroup.crosswalk || ObjectGroup.caution => s.surfaceConfidence,
          ObjectGroup.trafficRed || ObjectGroup.trafficGreen => s.trafficConfidence,
          _ => s.confidence,
        };
        if (r.confidence < minConf) continue;
      }
      final nb = r.normalizedBox;
      if (nb.width > 0.05 && nb.height > 0.05) {
        // boundingBox is in camera pixels and normalizedBox in 0..1, so their ratio gives the frame size.
        final iw = r.boundingBox.width / nb.width;
        final ih = r.boundingBox.height / nb.height;
        if (iw > 0 && ih > 0) {
          final aspect = iw / ih;
          if (aspect > 0.3 && aspect < 3.5) _imageAspect = aspect;
        }
      }
      out.add(Detection(
        group: group,
        label: r.className,
        confidence: r.confidence,
        imageBox: nb,
        viewBox: ViewMapper.imageToView(nb, _imageAspect, _viewSize),
      ));
    }
    return out;
  }

  // ------------------------------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------------------------------
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state != AppLifecycleState.resumed) {
      _audio.beepInterval = null;
      _engine?.reset();
      _marker.reset();
      _markerState = null;
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _accel?.cancel();
    _gyro?.cancel();
    _blinkTimer?.cancel();
    _audio.dispose();
    WakelockPlus.disable();
    super.dispose();
  }

  // ------------------------------------------------------------------------------------------
  // UI
  // ------------------------------------------------------------------------------------------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: LayoutBuilder(
        builder: (context, constraints) {
          _viewSize = constraints.biggest;
          return Stack(
            fit: StackFit.expand,
            children: [
              if (_yoloView != null) _yoloView! else const Center(child: CircularProgressIndicator()),
              IgnorePointer(
                child: CustomPaint(
                  painter: GuideOverlayPainter(
                    frame: _frame,
                    blinkOn: _blinkOn,
                    marker: _markerState,
                    debugDetections: kShowDetections ? _rawDetections : null,
                  ),
                ),
              ),
              if (_error != null) _buildError(_error!),
            ],
          );
        },
      ),
    );
  }

  /// Horizontal field of view of the screen image: the 4:3 camera frame fills the full screen width in
  /// landscape, so it is the camera's horizontal FOV (from the vertical FOV setting and frame aspect).
  double _horizontalFovDeg(GuideSettings? s) {
    final v = (s?.verticalFovDeg ?? 52) * math.pi / 360;
    return 2 * math.atan(math.tan(v) * _imageAspect) * 180 / math.pi;
  }

  Widget _buildError(String text) {
    return Center(
      child: Container(
        margin: const EdgeInsets.all(24),
        padding: const EdgeInsets.all(16),
        color: Colors.black87,
        child: Text(text, style: const TextStyle(color: Colors.redAccent, fontSize: 16)),
      ),
    );
  }
}
