import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Provider extends User{
	public Location location;
	public String[] opening_time;
	public List<Provider> partners;
	public List<Provider> applied_partners;
	public Map<Bike,Integer> registered_bikes;
	private List<Order> orders_record;
	private Map<Order, String> change_daterange_requests;
		
	public Provider(String userID,String password,String firstname,String surname,
					String phone_number,String email,String address,String postcode,String[] opening_time) {
		super(userID,password,firstname,surname,phone_number,email);
		this.partners=new ArrayList<>();
		this.applied_partners=new ArrayList<>();
		this.orders_record=new ArrayList<>();
		this.change_daterange_requests=new HashMap<>();
		this.location=new Location(postcode,address);
		this.opening_time=opening_time;
		this.registered_bikes=new HashMap<>();
		
	}
	public void setLocation(Location location) {
		this.location=location;		
	}
	
	public void registerBike(Bike bike) {
		this.registered_bikes.put(bike,1);					
	}
	
	public void updateBike(Bike bike) {
		for(Bike b:registered_bikes.keySet()) {
			if(b.equals(bike)) {
				int i=registered_bikes.get(bike);
				registered_bikes.put(bike, i+1);
			}
		}
	}
	public void apply_partnership(Provider provider) {
		if(!Database.registered_provider.isEmpty()) {
			for(Provider p:Database.registered_provider) {
				if(p.equals(provider)) {
					p.applied_partners.add(provider);
				}
			}
		}		
	}
	public void accept_partnership(Provider provider) {
		if(!applied_partners.isEmpty()&&!Database.registered_provider.isEmpty()) {
			for(Provider p:applied_partners) {
				if(p.equals(provider)) {
					applied_partners.remove(p);
					for(Provider p1:Database.registered_provider) {
						if(p1.equals(provider)) {
							p1.partners.add(this);
							partners.add(p1);
							break;
						}
					}
				}
			}
		}		
	}
	public void cancel_partnership(Provider provider) {
		if(!partners.isEmpty()) {
			partners.remove(provider);
		}
	}
	public void record_returned_bikes(Order order) {
		order.status="complete";
	}
}
