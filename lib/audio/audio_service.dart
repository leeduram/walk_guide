import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../logic/guide_engine.dart';

/// Beeps and voice messages, fully decoupled from the camera/inference loop.
///
/// The engine only sets [beepInterval]; a lightweight timer here decides when to beep, and the
/// actual playback happens natively, so slow audio can never delay frame processing.
class AudioService {
  AudioService();

  static const voiceFiles = {
    GuideEvent.obstacleOnPath: 'audio/obstacle_on_path.mp3', // 경로에 장애물이 있습니다
    GuideEvent.brailleLost: 'audio/braille_lost.mp3', // 점자블록이 끊겼습니다
    GuideEvent.roadAhead: 'audio/road.mp3', // 차도입니다
    GuideEvent.crosswalk: 'audio/crosswalk.mp3', // 횡단보도입니다
    GuideEvent.signalRed: 'audio/signal_red.mp3', // 빨간불입니다
    GuideEvent.signalGreen: 'audio/signal_green.mp3', // 초록불입니다
    GuideEvent.caution: 'audio/caution.mp3', // 발밑을 조심하세요
  };

  // Higher number wins when two messages compete (a higher one interrupts a lower one).
  // Crosswalk is above the signal so "횡단보도입니다" is not cut off by the signal that follows it.
  static const _priority = {
    GuideEvent.roadAhead: 7,
    GuideEvent.obstacleOnPath: 6,
    GuideEvent.crosswalk: 5,
    GuideEvent.signalRed: 4,
    GuideEvent.signalGreen: 4,
    GuideEvent.caution: 3,
    GuideEvent.brailleLost: 1,
  };

  final AudioPlayer _beep = AudioPlayer(playerId: 'walk_guide_beep');
  final AudioPlayer _voice = AudioPlayer(playerId: 'walk_guide_voice');
  final Set<GuideEvent> _availableVoices = {};
  final List<GuideEvent> _queue = [];

  Timer? _ticker;
  StreamSubscription<void>? _voiceDone;
  bool _ready = false;
  bool _beepAvailable = false;
  GuideEvent? _speaking;
  DateTime _speakingSince = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastBeep = DateTime.fromMillisecondsSinceEpoch(0);

  Duration? beepInterval;
  bool beepEnabled = true;
  bool voiceEnabled = true;

  /// Which voice files are missing (shown on screen so they can be added later).
  List<String> get missingVoiceFiles => [
        for (final e in voiceFiles.entries)
          if (!_availableVoices.contains(e.key)) 'assets/${e.value}',
      ];

  Future<void> init() async {
    final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final assets = manifest.listAssets().toSet();
    for (final e in voiceFiles.entries) {
      if (assets.contains('assets/${e.value}')) _availableVoices.add(e.key);
    }

    // Mix with other audio so the beep and the voice never steal audio focus from each other.
    final ctx = AudioContextConfig(focus: AudioContextConfigFocus.mixWithOthers).build();
    try {
      await _beep.setAudioContext(ctx);
      await _voice.setAudioContext(ctx);
      await _beep.setReleaseMode(ReleaseMode.stop);
      await _voice.setReleaseMode(ReleaseMode.stop);
      if (assets.contains('assets/audio/beep.wav')) {
        await _beep.setSource(AssetSource('audio/beep.wav'));
        _beepAvailable = true;
      }
    } catch (e) {
      debugPrint('AudioService init failed: $e');
    }

    _voiceDone = _voice.onPlayerComplete.listen((_) {
      _speaking = null;
      _playNextVoice();
    });
    _ticker = Timer.periodic(const Duration(milliseconds: 25), (_) => _tick());
    _ready = true;
  }

  void _tick() {
    final interval = beepInterval;
    if (!_ready || !_beepAvailable || !beepEnabled || interval == null) return;
    final now = DateTime.now();
    if (_speaking != null) {
      // Never beep over a voice message; recover if a completion event was missed.
      if (now.difference(_speakingSince) < const Duration(seconds: 6)) return;
      _speaking = null;
    }
    if (now.difference(_lastBeep) < interval) return;
    _lastBeep = now;
    unawaited(_playBeep());
  }

  Future<void> _playBeep() async {
    try {
      await _beep.stop();
      await _beep.resume();
    } catch (e) {
      debugPrint('beep failed: $e');
    }
  }

  /// Queue a voice message. A higher-priority message interrupts a lower-priority one.
  void speak(GuideEvent event) {
    if (!_ready || !voiceEnabled || !_availableVoices.contains(event)) return;
    if (_speaking == event || _queue.contains(event)) return;
    final current = _speaking;
    if (current != null && _priority[event]! > _priority[current]!) {
      _queue.insert(0, event);
      unawaited(_voice.stop().then((_) {
        _speaking = null;
        _playNextVoice();
      }));
      return;
    }
    _queue.add(event);
    _queue.sort((a, b) => _priority[b]!.compareTo(_priority[a]!));
    if (_speaking == null) _playNextVoice();
  }

  void _playNextVoice() {
    if (_queue.isEmpty || _speaking != null) return;
    final next = _queue.removeAt(0);
    _speaking = next;
    _speakingSince = DateTime.now();
    unawaited(() async {
      try {
        await _beep.stop();
        await _voice.stop();
        await _voice.play(AssetSource(voiceFiles[next]!));
      } catch (e) {
        debugPrint('voice failed: $e');
        _speaking = null;
      }
    }());
  }

  Future<void> dispose() async {
    _ticker?.cancel();
    await _voiceDone?.cancel();
    await _beep.dispose();
    await _voice.dispose();
  }
}
