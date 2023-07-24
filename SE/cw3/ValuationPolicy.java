
import java.math.BigDecimal;
import java.time.LocalDate;

public interface ValuationPolicy {
    public BigDecimal calculateValue(Bike bike, LocalDate date);
}
