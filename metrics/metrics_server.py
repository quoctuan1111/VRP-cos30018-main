from flask import Flask, jsonify
from src.protocols.communication_manager import CommunicationManager
from src.agents.master_routing_agent import MasterRoutingAgent
from src.agents.delivery_agent import DeliveryAgent
from src.protocols.message_protocol import Message, MessageType
import threading
import time

app = Flask(__name__)

print("Initializing Communication Manager...")
comm_manager = CommunicationManager()

# Create and register test agents
print("Creating test agents...")
mra = MasterRoutingAgent("MRA_1")
da1 = DeliveryAgent("DA_1", capacity=100, max_distance=1000)
da2 = DeliveryAgent("DA_2", capacity=150, max_distance=1200)

print("Registering agents...")
comm_manager.register_agent(mra)
comm_manager.register_agent(da1)
comm_manager.register_agent(da2)

def run_comm_manager():
    print("Starting Communication Manager...")
    comm_manager.start()

@app.route('/')
def home():
    return "VRP Metrics Server Running"

@app.route('/api/metrics')
def get_metrics():
    metrics = comm_manager.get_performance_metrics()
    print("Metrics requested:", metrics)
    return jsonify(metrics)

@app.route('/api/test/generate-load')
def generate_test_load():
    print("Generating test load...")
    for i in range(10):
        message = Message(
            msg_type=MessageType.CAPACITY_REQUEST,
            sender_id="MRA_1",
            receiver_id="DA_1",
            content={}
        )
        comm_manager.process_message(message)
        print(f"Processed message {i+1}/10")
    return jsonify({"status": "generated 10 test messages"})

if __name__ == '__main__':
    print("Starting VRP Metrics Server...")
    
    # Start communication manager in a separate thread
    comm_thread = threading.Thread(target=run_comm_manager)
    comm_thread.daemon = True
    comm_thread.start()
    
    # Wait a moment for the comm manager to initialize
    time.sleep(1)
    
    print("Starting Flask server...")
    app.run(debug=True, port=5000)