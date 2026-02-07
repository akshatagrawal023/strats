from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import os

app = Flask(__name__)  # Use default static folder
socketio = SocketIO(app, cors_allowed_origins="*")

# Cache latest payload for new connections
latest_greeks = None

@app.route('/')
def dashboard():
    return render_template('enhanced_dashboard.html')

@app.route('/simple')
def simple_dashboard():
    return render_template('dashboard.html')

@socketio.on('connect')
def handle_connect(auth=None):
    print('Client connected')
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
    print("Enhanced dashboard: http://localhost:5000")
    print("Simple dashboard: http://localhost:5000/simple")