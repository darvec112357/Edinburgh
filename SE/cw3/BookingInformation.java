import java.util.List;
import java.util.ArrayList;

public class BookingInformation {
	public List<Bike> required_bikes;
	public DateRange required_date;
	public Location location;
	public String collection_method;
	public String card;
	
	public BookingInformation(List<Bike> bikes,DateRange date,Location location,
							  String collect_method,String card) {
		this.location=location;
		this.card=card;
		this.required_bikes=bikes;
		this.required_date=date;
		this.collection_method=collect_method;
	}
	
}
