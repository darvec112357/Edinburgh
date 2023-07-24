import java.util.*;

public class Database {
	public static List<Provider> registered_provider;
	public List<Customer> registered_customer;
	public List<Bike> registered_bike;
	public List<Order> orders;
	public Map<Bike,Provider>satisfied_quotes;
	public List<DeliveryDriver> recorded_delivery_dirver;
	
	public boolean verify_booking_information(BookingInformation book_inf) {
		return true;
	}
	
	public Map<Bike,Provider> show_quotes(){
		return this.satisfied_quotes;
	}
	
	public void add_quote(Bike b, Provider p){
		this.satisfied_quotes.put(b,p);
	}
	
	public void update_order_status(Order o) {
		o.status = "compeleted ! ";
	}
	
	public void change_credit(User u) {
		u.credit -= 5;
	}
	public void showquotes(Map<Bike,Provider> map) {
		
	}

}

