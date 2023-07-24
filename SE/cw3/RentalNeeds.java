import java.util.*;

public class RentalNeeds {
	public BikeType required_bike_type;
	public int required_bike_num;
	public DateRange required_date;
	public Location location;
	
	public RentalNeeds(BikeType bike_type, int num,DateRange required_date, Location location) {
		this.required_bike_type=bike_type;
		
		this.required_date = required_date;
		this.location = location;
	}
	public boolean available_within_three_days(DateRange a,DateRange b) {
		DateRange s=new DateRange(a.getStart(),b.getStart());
		DateRange e=new DateRange(b.getEnd(),a.getEnd());
		return(3<=s.toDays()&&3<=e.toDays());
	}
	public void get_quote(){
		List providerlist = Database.registered_provider;
		for(int i =0; i < providerlist.size(); i++) {
			Provider p = (Provider)providerlist.get(i);
			if(this.location.isNearTo(p.location)) {
				Map<Bike,Integer> bikemap = p.registered_bikes;
				for(Bike b:bikemap.keySet()) {
					if(b.bike_type.equals(requires_bike_type)) {
						a
					}
				}
								
			}
		}
	} 
}
