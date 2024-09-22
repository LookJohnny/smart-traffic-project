import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class FlaskClient {

   public static int sendPredictionRequest(float[] sensorData) throws Exception {
      // 设置 Flask 服务器 URL
      String urlString = "http://localhost:5000/predict";
      URL url = new URL(urlString);
      HttpURLConnection conn = (HttpURLConnection) url.openConnection();
      conn.setRequestMethod("POST");
      conn.setRequestProperty("Content-Type", "application/json; utf-8");
      conn.setRequestProperty("Accept", "application/json");
      conn.setDoOutput(true);

      // 构建 JSON 对象
      JSONObject jsonInput = new JSONObject();
      jsonInput.put("data", sensorData);

      // 发送请求
      try(OutputStream os = conn.getOutputStream()) {
         byte[] input = jsonInput.toString().getBytes("utf-8");
         os.write(input, 0, input.length);
      }

      // 接收响应
      try(BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"))) {
         StringBuilder response = new StringBuilder();
         String responseLine = null;
         while ((responseLine = br.readLine()) != null) {
            response.append(responseLine.trim());
         }
         // 解析 JSON 响应
         JSONObject jsonResponse = new JSONObject(response.toString());
         int prediction = jsonResponse.getInt("prediction");
         return prediction;
      }
   }
}

