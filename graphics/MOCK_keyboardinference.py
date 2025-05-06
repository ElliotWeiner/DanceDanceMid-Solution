import random
import time
import pygame
import sys
import socket
import json


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
        print("Waiting for listener to connect...")

        # Accept a connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with: {client_address}")

        # Initialize pygame for keyboard handling
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Input Detection")

        current_direction = 4  # Start with no input
        confidence = 0.95  # Higher confidence for keyboard inputs
        counter = 0  # Message counter

        # Font initialization
        font = pygame.font.Font(None, 36)

        # Instructions text
        instructions = [
            "Press arrow keys to send direction commands",
            "UP: ↑  DOWN: ↓  LEFT: ←  RIGHT: →",
            "NONE: Space   QUIT: Esc",
        ]

        last_key_time = 0
        key_hold_threshold = 500  # ms
        key_state = {
            pygame.K_UP: False,
            pygame.K_DOWN: False,
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False,
            pygame.K_SPACE: False,
        }

        while True:
            key_pressed = False
            current_time = pygame.time.get_ticks()

            # Check for events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    pygame.quit()
                    print("\nDetection stream ended.")
                    return

                # Key down events
                elif event.type == pygame.KEYDOWN:
                    key_pressed = True
                    last_key_time = current_time

                    if event.key == pygame.K_UP:
                        current_direction = 0
                        key_state[pygame.K_UP] = True
                    elif event.key == pygame.K_DOWN:
                        current_direction = 1
                        key_state[pygame.K_DOWN] = True
                    elif event.key == pygame.K_LEFT:
                        current_direction = 2
                        key_state[pygame.K_LEFT] = True
                    elif event.key == pygame.K_RIGHT:
                        current_direction = 3
                        key_state[pygame.K_RIGHT] = True
                    elif event.key == pygame.K_SPACE:
                        current_direction = 4
                        key_state[pygame.K_SPACE] = True

                # Key up events
                elif event.type == pygame.KEYUP:
                    if event.key in key_state:
                        key_state[event.key] = False

            # Check for held keys
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] and key_state[pygame.K_UP]:
                current_direction = 0
                key_pressed = True
            elif keys[pygame.K_DOWN] and key_state[pygame.K_DOWN]:
                current_direction = 1
                key_pressed = True
            elif keys[pygame.K_LEFT] and key_state[pygame.K_LEFT]:
                current_direction = 2
                key_pressed = True
            elif keys[pygame.K_RIGHT] and key_state[pygame.K_RIGHT]:
                current_direction = 3
                key_pressed = True
            elif keys[pygame.K_SPACE] and key_state[pygame.K_SPACE]:
                current_direction = 4
                key_pressed = True

            # Auto-reset to NONE if no key is pressed for a while
            if not key_pressed and current_time - last_key_time > key_hold_threshold:
                if current_direction != 4:  # Only reset if not already NONE
                    current_direction = 4
                    key_pressed = True  # Trigger message send

            # Current timestamp
            timestamp = time.strftime("%H:%M:%S.%f")[:-3]

            # Direction name
            direction_name = get_direction_name(current_direction)

            # Create JSON message to send (only when key state changes)
            if key_pressed:
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

                    # Display current detection in console
                    conf_str = f"{confidence:.2f}"
                    print(f"| {direction_name:<10} | {conf_str:<10} | {timestamp} |")
                except:
                    print("Connection lost. Waiting for new connection...")
                    client_socket.close()
                    client_socket, client_address = server_socket.accept()
                    print(f"New connection established with: {client_address}")

            # Update screen with direction
            screen.fill((0, 0, 0))

            # Draw current direction
            text = font.render(f"Direction: {direction_name}", True, (255, 255, 255))
            screen.blit(text, (50, 80))

            # Draw confidence
            conf_text = font.render(
                f"Confidence: {confidence:.2f}", True, (255, 255, 255)
            )
            screen.blit(conf_text, (50, 120))

            # Draw message counter
            info_text = font.render(f"Messages sent: {counter}", True, (200, 200, 200))
            screen.blit(info_text, (50, 160))

            # Draw instructions
            small_font = pygame.font.Font(None, 24)
            for i, line in enumerate(instructions):
                instruction_text = small_font.render(line, True, (170, 170, 200))
                screen.blit(instruction_text, (50, 200 + i * 25))

            pygame.display.flip()

            # Short sleep to avoid overwhelming the CPU
            time.sleep(0.05)

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
