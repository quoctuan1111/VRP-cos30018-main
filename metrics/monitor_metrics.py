import requests
import time

def print_metrics():
    try:
        # First generate some test load
        requests.get("http://localhost:5000/api/test/generate-load")
        
        # Then get metrics
        response = requests.get("http://localhost:5000/api/metrics")
        metrics = response.json()
        
        print("\n=== Performance Metrics ===")
        print("Message Processing:")
        print(f"  Average Time: {metrics['message_processing']['average_time']:.2f}ms")
        
        print("\nMessage Types:")
        for msg_type, time in metrics['message_processing']['by_type'].items():
            print(f"  {msg_type}: {time:.2f}ms")
            
        print("\nQueue Processing:")
        print(f"  Current Rate: {metrics['queue_processing']['current_rate']:.2f} msgs/sec")
        print(f"  Average Rate: {metrics['queue_processing']['average_rate']:.2f} msgs/sec")
        print(f"  Peak Rate: {metrics['queue_processing']['peak_rate']:.2f} msgs/sec")
        
        print("\nMemory Usage:")
        print(f"  Current: {metrics['memory_usage']['statistics']['rss']['current']:.2f} MB")
        print(f"  Peak: {metrics['memory_usage']['statistics']['rss']['peak']:.2f} MB")
        
    except requests.exceptions.ConnectionError:
        print("Failed to connect to server. Is it running?")
    except Exception as e:
        print(f"Error: {e}")

def monitor_continuously(interval=5):
    print("Starting performance monitoring (Press Ctrl+C to stop)...")
    while True:
        print_metrics()
        time.sleep(interval)

if __name__ == "__main__":
    try:
        monitor_continuously()
    except KeyboardInterrupt:
        print("\nStopping monitoring...")