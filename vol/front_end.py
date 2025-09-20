from flask import Flask, render_template
from flask_socketio import SocketIO
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Cache latest payload for new connections
latest_greeks = None

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send latest snapshot if available
    global latest_greeks
    if latest_greeks is not None:
        socketio.emit('greeks_update', latest_greeks)

def broadcast_greeks(data):
    global latest_greeks
    latest_greeks = data
    socketio.emit('greeks_update', data)

def run_web_dashboard():
    """Run web dashboard in separate thread"""
    def run_server():
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("Dashboard available at: http://localhost:5000")