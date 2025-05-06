import random
import time
import pygame
import sys
import socket
import json


def mock_inference():
    """
    Mock inference function that simulates a model output.
    Returns:
    - 0: up
    - 1: down
    - 2: left
    - 3: right
    - 4: no input
    """
    # Simulate different inputs with higher probability of "no input"
    choices = [0, 1, 2, 3, 4, 4, 4, 4]
    return random.choice(choices)


def get_direction_name(direction_code):
    """Convert direction code to human-readable name"""
    directions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "NONE"}
    return directions.get(direction_code, "UNKNOWN")


def main():
    """Main function to stream detection results"""
    print("Starting input detection stream. Press Ctrl+C to exit.")
    print("=" * 50)
    print("| DETECTED INPUT | CONFIDENCE | TIMESTAMP |")
    print("=" * 50)

    # Initialize socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "localhost"
    port = 12345

    try:
        # Bind socket to address and port
        server_socket.bind((host, port))
        # Listen for incoming connections
        server_socket.listen(1)
        print(f"Publisher listening on {host}:{port}")

        # Accept a connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with: {client_address}")

        # Initialize pygame for keyboard handling
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Input Detection")

        current_direction = 4  # Start with no input
        confidence = 0.0
        counter = 0  # Message counter

        while True:
            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print("\nDetection stream ended.")
                    return
                # Allow manual direction setting with arrow keys for testing
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        current_direction = 0
                        confidence = 0.9
                    elif event.key == pygame.K_DOWN:
                        current_direction = 1
                        confidence = 0.9
                    elif event.key == pygame.K_LEFT:
                        current_direction = 2
                        confidence = 0.9
                    elif event.key == pygame.K_RIGHT:
                        current_direction = 3
                        confidence = 0.9
                    elif event.key == pygame.K_SPACE:
                        current_direction = 4
                        confidence = 0.9

            # Get mock inference result if no manual input
            if random.random() > 0.7:  # Only change sometimes to simulate stability
                if confidence > 0.3:  # Reduce confidence gradually
                    confidence -= 0.1
                else:
                    current_direction = mock_inference()
                    confidence = random.uniform(0.7, 0.99)

            # Current timestamp
            timestamp = time.strftime("%H:%M:%S.%f")[:-3]

            # Direction name
            direction_name = get_direction_name(current_direction)

            # Create JSON message to send
            counter += 1
            message = {
                "counter": counter,
                "direction_code": current_direction,
                "direction": direction_name,
                "confidence": round(confidence, 2),
                "timestamp": timestamp,
                "message": f"{direction_name} (conf: {confidence:.2f})",
            }

            # Send message to client
            try:
                json_message = json.dumps(message) + "\n"
                client_socket.sendall(json_message.encode("utf-8"))
            except:
                print("Connection lost. Waiting for new connection...")
                client_socket.close()
                client_socket, client_address = server_socket.accept()
                print(f"New connection established with: {client_address}")

            # Display current detection in console
            conf_str = f"{confidence:.2f}"
            print(
                f"| {direction_name:<10} | {conf_str:<10} | {timestamp} |",
                end="\r",
                flush=True,
            )

            # Update screen with direction
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 36)
            text = font.render(
                f"Direction: {direction_name} ({conf_str})", True, (255, 255, 255)
            )
            info_text = font.render(f"Message #{counter} sent", True, (200, 200, 200))
            screen.blit(text, (50, 120))
            screen.blit(info_text, (50, 170))
            pygame.display.flip()

            # Short sleep to avoid overwhelming the CPU
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nDetection stream ended by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        try:
            client_socket.close()
        except:
            pass
        server_socket.close()
        pygame.quit()


if __name__ == "__main__":
    main()
