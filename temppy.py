from flask import Flask, request, jsonify, render_template
import logging
from flask_cors import CORS  # Import CORS

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store LED states
led_states = {i: False for i in range(1, 7)}  # LEDs 1-6, initially off

@app.route('/status')
def get_status():
    """Get current status of all LEDs"""
    return jsonify(led_states)

@app.route('/control')
def control():
    """API endpoint to control LEDs"""
    command = request.args.get('command')
    if not command or len(command) < 2:
        return jsonify({"error": "Invalid command"}), 400
    
    try:
        led_number = int(command[0])
        state = command[1] == '1'
        
        if led_number not in range(1, 7):
            return jsonify({"error": "Invalid LED number"}), 400
        
        # Update LED state
        led_states[led_number] = state
        logger.info(f"LED {led_number} set to {state}")
        
        return jsonify({
            "success": True,
            "led": led_number,
            "state": state
        })
    except ValueError:
        return jsonify({"error": "Invalid command format"}), 400

@app.route('/control/all')
def control_all():
    """Control all LEDs at once"""
    state = request.args.get('state')
    if state not in ['0', '1']:
        return jsonify({"error": "Invalid state"}), 400
    
    new_state = state == '1'
    for led in led_states:
        led_states[led] = new_state
    
    logger.info(f"All LEDs set to {new_state}")
    return jsonify({"success": True, "state": new_state})

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

if __name__ == "__main__":
    try:
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise