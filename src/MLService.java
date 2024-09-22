import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.json.JSONObject;

@Service
public class MLService {

   private final RestTemplate restTemplate = new RestTemplate();

   public String predictDangerousDriving(double accX, double accY, double accZ,
                                         double gyroX, double gyroY, double gyroZ) {
      String url = "http://localhost:5000/predict";  // Flask服务运行的地址

      JSONObject request = new JSONObject();
      request.put("AccX", accX);
      request.put("AccY", accY);
      request.put("AccZ", accZ);
      request.put("GyroX", gyroX);
      request.put("GyroY", gyroY);
      request.put("GyroZ", gyroZ);

      String response = restTemplate.postForObject(url, request.toString(), String.class);
      JSONObject jsonResponse = new JSONObject(response);

      return jsonResponse.getString("prediction");
   }
}
