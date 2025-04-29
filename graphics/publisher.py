#!/usr/bin/env python3
import socket
import time
import json

def start_publisher():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Define host and port
    host = 'localhost'
    port = 12345
    
    # Bind the socket to the address
    server_socket.bind((host, port))
    
    # Listen for incoming connections (max 1 connection in queue)
    server_socket.listen(1)
    print(f"Publisher started on {host}:{port}")
    
    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address}")
    
    try:
        # Send data periodically
        counter = 0
        while True:
            # Create data to send
            data = {
                "timestamp": time.time(),
                "counter": counter,
                "message": f"Hello from publisher! Count: {counter}"
            }
            
            # Convert to JSON string and encode to bytes
            json_data = json.dumps(data) + "\n"  # Add newline as a message delimiter
            client_socket.send(json_data.encode('utf-8'))
            
            print(f"Sent: {data}")
            counter += 1
            
            # Sleep for a second before sending next data
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Publisher terminated by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the sockets
        client_socket.close()
        server_socket.close()
        print("Publisher shutdown")

if __name__ == "__main__":
    start_publisher()