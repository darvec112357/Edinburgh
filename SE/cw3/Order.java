import java.util.Map;

public class Order {
	public String order_num;
	public String customer;
	public double total_price;
	public String payment_method;
	public String collect_method;
	public String status;
	public Map<Bike,Integer> information_bikes;
	public DateRange date_range;
	
	public Order(String num,String customer,double price,String payment_method,String collect_method,
			     String status,Map<Bike,Integer> info, DateRange date) {
		
		this.collect_method=collect_method;
		this.customer=customer;
		this.total_price=price;
		this.payment_method=payment_method;
		this.status=status;
		this.information_bikes=info;
		this.date_range=date;
		this.order_num=num;
		
	}
}
