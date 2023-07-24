
public class User {
	public String userID;
	protected String password;
	protected String firstname;
	public String surname;
	protected String phone_number;
	protected String email;
	public int age_of_account;
	public int credit;
	
	public User(String userID,String password,String firstname,String surname,String phone_number,String email) {
		this.userID=userID;
		this.password=password;
		this.firstname=firstname;
		this.surname=surname;
		this.phone_number=phone_number;
		this.email=email;
		this.credit=100;
	}
    
	protected void register_customer() {
		
	}

}
