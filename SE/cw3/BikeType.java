
import java.math.BigDecimal;
import java.util.Objects;

public class BikeType {
	public final String utility;
	public final double max_speed;
	public final double wheel_radius;
	public final String material;
	public final double full_replacement_value;
	
	public BikeType(double max_speed,double wheel_radius,String material,double full_replacement_value,String utility) {		
		this.utility=utility;
		this.wheel_radius=wheel_radius;
		this.material=material;
		this.max_speed=max_speed;
		this.full_replacement_value=full_replacement_value;
		
	}
    public double getReplacementValue() {   
        return full_replacement_value;
    }
}