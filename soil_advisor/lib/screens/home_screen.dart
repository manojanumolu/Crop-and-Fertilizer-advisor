import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _imageFile;
  bool _loading = false;

  // Chemical sliders
  double _n  = 90;
  double _p  = 42;
  double _k  = 43;
  double _ph = 6.5;

  // Environmental
  final _tempCtrl = TextEditingController(text: '25.0');
  final _humCtrl  = TextEditingController(text: '80.0');
  final _rainCtrl = TextEditingController(text: '200.0');

  // Farm history
  final _yldCtrl  = TextEditingController(text: '2500.0');
  final _fertCtrl = TextEditingController(text: '120.0');

  // Farm details
  String _season = 'Kharif';
  String _irrig  = 'Canal';
  String _prev   = 'Wheat';
  String _region = 'South';

  final _formKey = GlobalKey<FormState>();

  static const Color _green      = Color(0xFF2E7D32);
  static const Color _darkGreen  = Color(0xFF1B5E20);
  static const Color _lightGreen = Color(0xFFE8F5E9);

  @override
  void dispose() {
    _tempCtrl.dispose();
    _humCtrl.dispose();
    _rainCtrl.dispose();
    _yldCtrl.dispose();
    _fertCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: source, imageQuality: 85);
    if (picked != null) setState(() => _imageFile = File(picked.path));
  }

  void _showImagePicker() {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: 8),
            Container(
              width: 40, height: 4,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 16),
            ListTile(
              leading: const CircleAvatar(
                backgroundColor: _lightGreen,
                child: Icon(Icons.photo_library, color: _green),
              ),
              title: const Text('Choose from Gallery'),
              onTap: () { Navigator.pop(context); _pickImage(ImageSource.gallery); },
            ),
            ListTile(
              leading: const CircleAvatar(
                backgroundColor: _lightGreen,
                child: Icon(Icons.camera_alt, color: _green),
              ),
              title: const Text('Take a Photo'),
              onTap: () { Navigator.pop(context); _pickImage(ImageSource.camera); },
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  Future<void> _analyze() async {
    if (_imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select a soil image first'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }
    if (!(_formKey.currentState?.validate() ?? false)) return;

    setState(() => _loading = true);
    try {
      final result = await ApiService.predict(
        imageFile: _imageFile!,
        n:      _n,
        p:      _p,
        k:      _k,
        temp:   double.tryParse(_tempCtrl.text) ?? 25.0,
        hum:    double.tryParse(_humCtrl.text)  ?? 80.0,
        rain:   double.tryParse(_rainCtrl.text) ?? 200.0,
        ph:     _ph,
        yld:    double.tryParse(_yldCtrl.text)  ?? 2500.0,
        fert:   double.tryParse(_fertCtrl.text) ?? 120.0,
        season: _season,
        irrig:  _irrig,
        prev:   _prev,
        region: _region,
      );
      if (mounted) {
        Navigator.pushNamed(context, '/results', arguments: result);
      }
    } catch (e) {
      if (mounted) {
        showDialog(
          context: context,
          builder: (_) => AlertDialog(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            title: const Text('Connection Error'),
            content: Text(
              'Could not connect to the API.\n\nMake sure api.py is running:\n'
              '  python api.py\n\nError: $e',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK', style: TextStyle(color: _green)),
              ),
            ],
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Widget _sectionHeader(IconData icon, String title) {
    return Row(
      children: [
        Icon(icon, color: _darkGreen, size: 22),
        const SizedBox(width: 8),
        Text(title, style: const TextStyle(
          fontSize: 16, fontWeight: FontWeight.bold, color: _darkGreen,
        )),
      ],
    );
  }

  Widget _sliderRow(String label, double value, double min, double max,
      ValueChanged<double> onChanged) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(
              fontSize: 13, color: _darkGreen, fontWeight: FontWeight.w500,
            )),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: _lightGreen,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(value.toStringAsFixed(1), style: const TextStyle(
                fontWeight: FontWeight.bold, color: _darkGreen, fontSize: 13,
              )),
            ),
          ],
        ),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: _green,
            thumbColor: _green,
            inactiveTrackColor: const Color(0xFFC8E6C9),
            overlayColor: const Color(0x292E7D32),
            trackHeight: 4,
          ),
          child: Slider(
            value: value, min: min, max: max,
            divisions: ((max - min) * 2).toInt().clamp(1, 200),
            onChanged: onChanged,
          ),
        ),
      ],
    );
  }

  Widget _textField(TextEditingController ctrl, String label,
      {String? suffix}) {
    return TextFormField(
      controller: ctrl,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      decoration: InputDecoration(labelText: label, suffixText: suffix),
      validator: (v) {
        if (v == null || v.isEmpty) return 'Required';
        if (double.tryParse(v) == null) return 'Enter a number';
        return null;
      },
    );
  }

  Widget _dropdown(String label, String value, List<String> items,
      ValueChanged<String?> onChanged) {
    return DropdownButtonFormField<String>(
      value: value,
      decoration: InputDecoration(labelText: label),
      items: items.map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
      onChanged: onChanged,
      style: const TextStyle(color: Colors.black87, fontSize: 15),
      dropdownColor: Colors.white,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true,
      appBar: AppBar(
        title: const Text('AgroLens',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF1B5E20),
      ),
      backgroundColor: const Color(0xFFF1F8E9),
      body: Form(
        key: _formKey,
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [

              // SECTION A — Soil Image
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _sectionHeader(Icons.camera_alt, 'Soil Image'),
                      const SizedBox(height: 16),
                      if (_imageFile != null) ...[
                        ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: Image.file(
                            _imageFile!,
                            height: 200,
                            width: double.infinity,
                            fit: BoxFit.cover,
                          ),
                        ),
                        const SizedBox(height: 12),
                      ],
                      SizedBox(
                        width: double.infinity,
                        child: OutlinedButton.icon(
                          onPressed: _showImagePicker,
                          icon: const Icon(Icons.add_photo_alternate, color: _green),
                          label: Text(
                            _imageFile == null ? 'Select Image' : 'Change Image',
                            style: const TextStyle(color: _green),
                          ),
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: _green),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            padding: const EdgeInsets.symmetric(vertical: 14),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              // SECTION B — Chemical Properties
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _sectionHeader(Icons.science, 'Chemical Properties'),
                      const SizedBox(height: 16),
                      _sliderRow('Nitrogen (N)', _n, 0, 140,
                          (v) => setState(() => _n = v)),
                      const SizedBox(height: 4),
                      _sliderRow('Phosphorus (P)', _p, 0, 145,
                          (v) => setState(() => _p = v)),
                      const SizedBox(height: 4),
                      _sliderRow('Potassium (K)', _k, 0, 205,
                          (v) => setState(() => _k = v)),
                      const SizedBox(height: 4),
                      _sliderRow('Soil pH', _ph, 3.5, 9.5,
                          (v) => setState(() => _ph = v)),
                    ],
                  ),
                ),
              ),

              // SECTION C — Environmental
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _sectionHeader(Icons.wb_sunny, 'Environmental Conditions'),
                      const SizedBox(height: 16),
                      _textField(_tempCtrl, 'Temperature', suffix: 'deg C'),
                      const SizedBox(height: 12),
                      _textField(_humCtrl, 'Humidity', suffix: '%'),
                      const SizedBox(height: 12),
                      _textField(_rainCtrl, 'Rainfall', suffix: 'mm'),
                    ],
                  ),
                ),
              ),

              // SECTION D — Farm History
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _sectionHeader(Icons.history, 'Farm History'),
                      const SizedBox(height: 16),
                      _textField(_yldCtrl, 'Previous Yield', suffix: 't/ha'),
                      const SizedBox(height: 12),
                      _textField(_fertCtrl, 'Fertilizer Used', suffix: 'kg/ha'),
                    ],
                  ),
                ),
              ),

              // SECTION E — Farm Details
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _sectionHeader(Icons.agriculture, 'Farm Details'),
                      const SizedBox(height: 16),
                      _dropdown('Season', _season,
                          ['Kharif', 'Rabi', 'Zaid'],
                          (v) => setState(() => _season = v!)),
                      const SizedBox(height: 12),
                      _dropdown('Irrigation', _irrig,
                          ['Canal', 'Drip', 'Rainfed', 'Sprinkler'],
                          (v) => setState(() => _irrig = v!)),
                      const SizedBox(height: 12),
                      _dropdown('Previous Crop', _prev,
                          ['Cotton', 'Maize', 'Potato', 'Rice',
                           'Sugarcane', 'Tomato', 'Wheat'],
                          (v) => setState(() => _prev = v!)),
                      const SizedBox(height: 12),
                      _dropdown('Region', _region,
                          ['Central', 'East', 'North', 'South', 'West'],
                          (v) => setState(() => _region = v!)),
                    ],
                  ),
                ),
              ),

              // ANALYZE BUTTON
              const SizedBox(height: 8),
              SizedBox(
                height: 56,
                child: ElevatedButton(
                  onPressed: _loading ? null : _analyze,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _green,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                    disabledBackgroundColor: Colors.grey[400],
                  ),
                  child: _loading
                      ? const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SizedBox(
                              width: 22, height: 22,
                              child: CircularProgressIndicator(
                                color: Colors.white, strokeWidth: 2.5,
                              ),
                            ),
                            SizedBox(width: 12),
                            Text('Analyzing...',
                                style: TextStyle(
                                    fontSize: 18, fontWeight: FontWeight.bold)),
                          ],
                        )
                      : const Text('Analyze Soil',
                          style: TextStyle(
                              fontSize: 18, fontWeight: FontWeight.bold)),
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}
