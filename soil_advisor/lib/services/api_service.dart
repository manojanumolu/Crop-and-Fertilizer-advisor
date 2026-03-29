import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  // Local Flask API (Android emulator uses 10.0.2.2 to reach host localhost)
  static const String baseUrl = 'http://10.0.2.2:8000';

  static Future<Map<String, dynamic>> predict({
    required File imageFile,
    required double n,
    required double p,
    required double k,
    required double temp,
    required double hum,
    required double rain,
    required double ph,
    required double yld,
    required double fert,
    required String season,
    required String irrig,
    required String prev,
    required String region,
  }) async {
    final uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      await http.MultipartFile.fromPath('image', imageFile.path),
    );

    request.fields['n']      = n.toString();
    request.fields['p']      = p.toString();
    request.fields['k']      = k.toString();
    request.fields['temp']   = temp.toString();
    request.fields['hum']    = hum.toString();
    request.fields['rain']   = rain.toString();
    request.fields['ph']     = ph.toString();
    request.fields['yld']    = yld.toString();
    request.fields['fert']   = fert.toString();
    request.fields['season'] = season;
    request.fields['irrig']  = irrig;
    request.fields['prev']   = prev;
    request.fields['region'] = region;

    final streamedResponse = await request.send().timeout(
      const Duration(seconds: 60),
    );
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return json.decode(response.body) as Map<String, dynamic>;
    } else {
      final err = json.decode(response.body);
      throw Exception(err['error'] ?? 'Server error ${response.statusCode}');
    }
  }
}
