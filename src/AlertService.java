import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

@Service
public class AlertService {

   private static final String TOPIC = "vehicle_alerts";

   @Autowired
   private KafkaTemplate<String, String> kafkaTemplate;

   public void sendAlert(String message) {
      kafkaTemplate.send(TOPIC, message);
   }
}

