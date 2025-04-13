import os
import time
import RPi.GPIO as GPIO

# Check if we're running on hardware that has GPIO capabilities
def is_gpio_available():
    try:
        import Adafruit_DHT
        # Try to check if gpio hardware is available
        if os.path.exists('/dev/gpiomem'):
            return True
        return False
    except ImportError:
        return False

def setup_gpio():
    """Set up GPIO if available"""
    if is_gpio_available():
        # Set GPIO mode
        GPIO.setmode(GPIO.BCM)
        
        # Define GPIO pins
        IR_PIN = 17
        
        # Set up GPIO pins
        GPIO.setup(IR_PIN, GPIO.IN)
        
        return IR_PIN
    return None

def read_dht_sensor():
    """
    Read temperature and humidity sensor data if available.
    Falls back to mock data if hardware or GPIO is not available.
    """
    if is_gpio_available():
        try:
            import Adafruit_DHT
            SENSOR = Adafruit_DHT.DHT11
            GPIO_PIN = 4  # Change if your sensor uses a different pin
            humidity, temperature = Adafruit_DHT.read_retry(SENSOR, GPIO_PIN)
            if humidity is not None and temperature is not None:
                return {
                    "temperature": round(temperature, 1),
                    "humidity": round(humidity, 1),
                    "status": "success"
                }
            else:
                return {
                    "temperature": 4.0,
                    "humidity": 65.0,
                    "status": "sensor readings failed, using defaults"
                }
        except Exception as e:
            return {
                "temperature": 4.0,
                "humidity": 65.0,
                "status": f"error: {str(e)}"
            }
    else:
        # Return mock data for non-GPIO environments (like desktop computers)
        return {
            "temperature": 4.0,
            "humidity": 65.0,
            "status": "simulated (GPIO hardware not available)"
        }

def check_ir_sensor(ir_pin):
    """Check if IR sensor detects signal"""
    if ir_pin is not None:
        return GPIO.input(ir_pin) == GPIO.LOW
    return False

def run_sensor_monitor():
    """Monitor both sensors together"""
    ir_pin = setup_gpio()
    
    print("Starting sensor monitoring...")
    print("Press CTRL+C to exit")
    
    try:
        while True:
            # Read DHT sensor
            dht_data = read_dht_sensor()
            print(f"Temperature: {dht_data['temperature']}°C, Humidity: {dht_data['humidity']}%")
            
            # Check IR sensor
            if check_ir_sensor(ir_pin):
                print("IR signal detected!")
            
            time.sleep(2)  # Wait before next reading
    except KeyboardInterrupt:
        if is_gpio_available():
            GPIO.cleanup()
        print("Monitoring stopped")

if __name__ == "__main__":
    run_sensor_monitor()
