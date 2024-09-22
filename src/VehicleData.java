import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;

@Setter
@Getter
public class VehicleData implements Serializable {
   // Getters and Setters
   private String vehicleId;
   private double speed;
   private double acceleration;
   private int hardBrakes;
   // Getters and Setters
   private double accX;
   private double accY;
   private double accZ;
   private double gyroX;
   private double gyroY;
   private double gyroZ;
   private String timestamp;

}

