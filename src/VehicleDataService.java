import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

@Service
public class VehicleDataService {

   @Autowired
   private RedisTemplate<String, VehicleData> redisTemplate;

   public void saveVehicleData(VehicleData vehicleData) {
      redisTemplate.opsForValue().set(vehicleData.getVehicleId(), vehicleData);
   }

   public VehicleData getVehicleData(String vehicleId) {
      return redisTemplate.opsForValue().get(vehicleId);
   }
}

