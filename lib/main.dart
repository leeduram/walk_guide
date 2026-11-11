import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'ui/guide_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Landscape: matches the 1920x1080 training images and a wider view of the path.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.landscapeLeft, DeviceOrientation.landscapeRight]);
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  runApp(const WalkGuideApp());
}

class WalkGuideApp extends StatelessWidget {
  const WalkGuideApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '보행 도우미',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFE5FF00), brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: const GuideScreen(),
    );
  }
}
