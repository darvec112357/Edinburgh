import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Customer extends User{
	private List<Order> order_history;	
	public Customer(String userID,String password,String firstname,String surname,String phone_number,String email) {
		super(userID,password,firstname,surname,phone_number,email);					
		this.order_history=new ArrayList<>();
	}	
	private Map<Bike,Provider> getquotes(List<String> strs){
		return null;
	}
	private Order bookquotes(BookingInformation info) {
		return null;
	}
	
}
