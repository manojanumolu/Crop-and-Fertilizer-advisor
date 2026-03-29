import 'package:flutter/material.dart';

class ResultsScreen extends StatelessWidget {
  final Map<String, dynamic> data;

  const ResultsScreen({super.key, required this.data});

  static const Map<String, Color> soilColors = {
    'Red Soil':      Color(0xFFC62828),
    'Black Soil':    Color(0xFF212121),
    'Alluvial Soil': Color(0xFF795548),
    'Clay Soil':     Color(0xFFFF8F00),
    'Laterite Soil': Color(0xFFBF360C),
    'Yellow Soil':   Color(0xFFF9A825),
    'Sandy Soil':    Color(0xFFD4A017),
  };

  static const Map<String, String> cropIcons = {
    'Cotton':       '🌿', 'Maize':      '🌽', 'Rice':       '🌾',
    'Wheat':        '🌾', 'Sugarcane':  '🎋', 'Potato':     '🥔',
    'Tomato':       '🍅', 'Groundnut':  '🥜', 'Soybean':    '🫘',
    'Sunflower':    '🌻', 'Barley':     '🌾', 'Mustard':    '🌼',
    'Chickpea':     '🫘', 'Watermelon': '🍉', 'Cucumber':   '🥒',
    'Pumpkin':      '🎃', 'Mango':      '🥭', 'Banana':     '🍌',
    'Cashew':       '🌰', 'Rubber':     '🌳', 'Tea':        '🍵',
    'Coffee':       '☕', 'Tapioca':    '🌿', 'Turmeric':   '🟡',
    'Ginger':       '🫚', 'Pineapple':  '🍍', 'Jackfruit':  '🍈',
    'Jute':         '🌿', 'Sorghum':    '🌾', 'Sesame':     '🌿',
    'Linseed':      '🌼', 'Safflower':  '🌼', 'Moong':      '🫘',
    'Taro':         '🌿', 'Spinach':    '🥬', 'Muskmelon':  '🍈',
  };

  @override
  Widget build(BuildContext context) {
    final soilName   = data['soil_name'] as String? ?? 'Unknown';
    final confidence = data['confidence'] as num? ?? 0;
    final allProbs   = Map<String, dynamic>.from(data['all_probs'] as Map? ?? {});
    final soilFert   = Map<String, dynamic>.from(data['soil_fert'] as Map? ?? {});
    final cropRecs   = List<Map<String, dynamic>>.from(
        (data['crop_recs'] as List? ?? []).map((e) => Map<String, dynamic>.from(e)));

    final soilColor = soilColors[soilName] ?? const Color(0xFF2E7D32);
    final sortedProbs = allProbs.entries.toList()
      ..sort((a, b) => (b.value as num).compareTo(a.value as num));

    return Scaffold(
      resizeToAvoidBottomInset: false,
      appBar: AppBar(
        title: const Text('Analysis Results',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF1B5E20),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      backgroundColor: const Color(0xFFF1F8E9),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [

            // ── SOIL TYPE CARD ─────────────────────────────────
            Container(
              decoration: BoxDecoration(
                color: soilColor,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: soilColor.withAlpha(100),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'DETECTED SOIL TYPE',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 12,
                      letterSpacing: 2,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    soilName,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.white.withAlpha(50),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      'Confidence: ${confidence.toStringAsFixed(1)}%',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // ── PROBABILITY BARS CARD ──────────────────────────
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.bar_chart, color: Color(0xFF1B5E20)),
                        SizedBox(width: 8),
                        Text('Soil Probability Breakdown',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF1B5E20),
                            )),
                      ],
                    ),
                    const SizedBox(height: 16),
                    ...sortedProbs.map((entry) {
                      final prob  = (entry.value as num).toDouble();
                      final color = soilColors[entry.key] ?? const Color(0xFF2E7D32);
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(entry.key,
                                    style: const TextStyle(
                                        fontSize: 13, color: Colors.black87)),
                                Text('${prob.toStringAsFixed(1)}%',
                                    style: TextStyle(
                                      fontSize: 13,
                                      fontWeight: FontWeight.bold,
                                      color: color,
                                    )),
                              ],
                            ),
                            const SizedBox(height: 4),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(6),
                              child: LinearProgressIndicator(
                                value: prob / 100,
                                backgroundColor: const Color(0xFFEEEEEE),
                                valueColor:
                                    AlwaysStoppedAnimation<Color>(color),
                                minHeight: 10,
                              ),
                            ),
                          ],
                        ),
                      );
                    }),
                  ],
                ),
              ),
            ),

            // ── RECOMMENDED CROPS CARD ─────────────────────────
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.grass, color: Color(0xFF1B5E20)),
                        SizedBox(width: 8),
                        Text('Recommended Crops',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF1B5E20),
                            )),
                      ],
                    ),
                    const SizedBox(height: 16),
                    ...cropRecs.map((crop) {
                      final name   = crop['name'] as String? ?? '';
                      final rank   = crop['rank']  as int? ?? 1;
                      final stars  = crop['stars'] as int? ?? 1;
                      final fert   = crop['fertilizer'] as String? ?? '';
                      final npk    = crop['npk']        as String? ?? '';
                      final icon   = cropIcons[name] ?? '🌱';
                      final starStr = '★' * stars + '☆' * (3 - stars);

                      return Container(
                        margin: const EdgeInsets.only(bottom: 10),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: rank == 1
                                ? const Color(0xFF2E7D32)
                                : const Color(0xFFC8E6C9),
                            width: rank == 1 ? 2 : 1,
                          ),
                        ),
                        padding: const EdgeInsets.all(12),
                        child: Row(
                          children: [
                            Text(icon, style: const TextStyle(fontSize: 32)),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(name,
                                      style: const TextStyle(
                                        fontWeight: FontWeight.bold,
                                        fontSize: 15,
                                        color: Color(0xFF1B5E20),
                                      )),
                                  const SizedBox(height: 2),
                                  Text(
                                    '$starStr  Rank #$rank',
                                    style: const TextStyle(
                                        fontSize: 13, color: Colors.grey),
                                  ),
                                  const SizedBox(height: 2),
                                  Text(
                                    '💊 $fert  |  📏 $npk',
                                    style: const TextStyle(
                                        fontSize: 12,
                                        color: Color(0xFF2E7D32)),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      );
                    }),
                  ],
                ),
              ),
            ),

            // ── FERTILIZER CARD ────────────────────────────────
            Container(
              decoration: BoxDecoration(
                color: const Color(0xFFFFF8E1),
                borderRadius: BorderRadius.circular(16),
                border: const Border(
                  left: BorderSide(color: Color(0xFFFF8F00), width: 4),
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.orange.withAlpha(30),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Row(
                    children: [
                      Icon(Icons.science, color: Color(0xFFE65100)),
                      SizedBox(width: 8),
                      Text('Fertilizer Recommendation',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFFE65100),
                          )),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      const Text('Type:  ',
                          style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Colors.black87)),
                      Expanded(
                        child: Text(
                          soilFert['fertilizer'] as String? ?? '',
                          style: const TextStyle(color: Colors.black87),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Row(
                    children: [
                      const Text('NPK:  ',
                          style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Colors.black87)),
                      Expanded(
                        child: Text(
                          soilFert['npk'] as String? ?? '',
                          style: const TextStyle(color: Colors.black87),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}
