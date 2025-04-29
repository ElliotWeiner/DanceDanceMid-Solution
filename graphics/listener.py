#!/usr/bin/env python3
import socket
import json

def start_listener():
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Define host and port to connect to
    host = 'localhost'
    port = 12345
    
    try:
        # Connect to the publisher
        print(f"Trying to connect to publisher at {host}:{port}...")
        client_socket.connect((host, port))
        print("Connected to publisher!")
        
        # Buffer for incomplete data
        buffer = ""
        
        # Receive and process data
        while True:
            # Receive data
            data = client_socket.recv(1024).decode('utf-8')
            
            # If no data, connection was closed
            if not data:
                print("Connection closed by publisher")
                break
            
            # Add received data to buffer
            buffer += data
            
            # Process complete messages (delimited by newlines)
            while "\n" in buffer:
                # Split at first newline
                message, buffer = buffer.split("\n", 1)
                
                try:
                    # Parse JSON message
                    parsed_data = json.loads(message)
                    
                    # Print only essential information for faster display
                    print(f"Received #{parsed_data['counter']}: {parsed_data['message']}")
                    
                    # Force flush the output buffer to ensure real-time printing
                    import sys
                    sys.stdout.flush()
                    
                except json.JSONDecodeError:
                    print(f"Failed to parse message: {message}")
    
    except KeyboardInterrupt:
        print("Listener terminated by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the socket
        client_socket.close()
        print("Listener shutdown")

if __name__ == "__main__":
    start_listener()