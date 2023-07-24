import java.util.*;

	public class DeliveryDriver extends User{

	public List<String> working_hour;

	public String company;

	public List<Order> orders;


	public DeliveryDriver(String userID, String password, String firstname, String surname, String phone_number,
		String email,List<String> working_hour,String company,List<Order> orders) {

		super(userID, password, firstname, surname, phone_number, email);
	 	this.working_hour=working_hour;
	 	this.company=company;
	 	this.orders=orders;
	}


	public void setWorking_hour(List<String> working_hour) {
		this.working_hour = working_hour;
	}

	public void setCompany(String company) {
		this.company = company;
	}

	public void setOrders(List<Order> orders) {
		this.orders = orders;
	}




}