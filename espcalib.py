import requests
import time
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESPCalibrator:
    def __init__(self, esp_urls: Dict[int, str], delay: int = 3):
        """
        Initialize the ESP calibrator
        
        Args:
            esp_urls (Dict[int, str]): Dictionary mapping ESP numbers to their URLs
            delay (int): Delay in seconds between switching ESPs (default: 3)
        """
        self.esp_urls = esp_urls
        self.delay = delay
        
    def turn_all_off(self):
        """Turn off all ESP LEDs"""
        logger.info("Turning off all ESPs...")
        for esp_num, url in self.esp_urls.items():
            try:
                requests.get(f"{url}/off", timeout=0.5)
                time.sleep(0.1)  # Small delay between commands
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to turn off ESP {esp_num}: {e}")

    def calibrate(self):
        """Run the calibration sequence"""
        try:
            # First turn all off
            self.turn_all_off()
            time.sleep(1)
            
            print("\n=== ESP Calibration Sequence ===")
            print("Watch the LEDs and note their positions...")
            print("Press Ctrl+C to stop the sequence at any time\n")
            
            while True:  # Loop continuously until interrupted
                for esp_num, url in sorted(self.esp_urls.items()):
                    try:
                        # Turn on current ESP
                        print(f"\nActivating ESP #{esp_num}")
                        print(f"URL: {url}")
                        print(f"Grid Position: {self._get_grid_position(esp_num)}")
                        print("If this matches the LED that's currently lit, note this mapping.")
                        
                        requests.get(f"{url}/on", timeout=7)
                        time.sleep(self.delay)
                        
                        # Turn off current ESP
                        requests.get(f"{url}/off", timeout=3)
                        time.sleep(0.5)  # Brief pause between ESPs
                        
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Error communicating with ESP {esp_num}: {e}")
                        continue
                
                print("\nSequence complete. Starting over...")
                print("Press Ctrl+C to exit\n")
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\nCalibration sequence stopped.")
            self.turn_all_off()
            print("All ESPs turned off.")
            
    def _get_grid_position(self, esp_num: int) -> str:
        """
        Convert ESP number to grid position
        Grid is 3x2 (rows x columns), numbered like this:
        1 2
        3 4
        5 6
        """
        row = (esp_num - 1) // 2
        col = (esp_num - 1) % 2
        return f"Row {row + 1}, Column {col + 1}"

def main():
    # Configuration (using the same ESP URLs from your main script)
    ESP_URLS = {
        1: "http://192.168.137.101",
        2: "http://192.168.137.102",
        3: "http://192.168.137.103",
        4: "http://192.168.137.104",
        5: "http://192.168.137.105",
        6: "http://192.168.137.106"
    }
    
    # Create and run calibrator
    calibrator = ESPCalibrator(ESP_URLS, delay=3)
    calibrator.calibrate()

if __name__ == "__main__":
    main()