import java.util.Map;

public class Bike {
	public BikeType bike_type;
	public String owner;
	public double deposit_rate;
	public double daily_rental_price;
	public Map<DateRange,Boolean> reserve_status;
	public boolean broken_status;
	public int starting_year;
	
	public Bike(BikeType bike_type,String owner, double deposit_rate, double daily_rental_price,
			int starting_year,boolean broken_status,Map<DateRange,Boolean> reserve_status) {
		this.bike_type=bike_type;
		this.owner=owner;
		this.deposit_rate=deposit_rate;
		this.daily_rental_price=daily_rental_price;
		this.starting_year=starting_year;
		this.broken_status=broken_status;
		this.reserve_status=reserve_status;
	}
	
    public BikeType getType() {
        return bike_type;
    }
}