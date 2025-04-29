# stream_detector.py
import random
import time
import pygame
import sys

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
    directions = {
        0: "UP    ",
        1: "DOWN  ",
        2: "LEFT  ",
        3: "RIGHT ",
        4: "NONE  "
    }
    return directions.get(direction_code, "UNKNOWN")

def main():
    """Main function to stream detection results"""
    print("Starting input detection stream. Press Ctrl+C to exit.")
    print("="*50)
    print("| DETECTED INPUT | CONFIDENCE | TIMESTAMP      |")
    print("="*50)
    
    try:
        # Initialize pygame for keyboard handling
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Input Detection")
        
        current_direction = 4  # Start with no input
        confidence = 0.0
        
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
            
            # Display current detection
            direction_name = get_direction_name(current_direction)
            conf_str = f"{confidence:.2f}"
            print(f"| {direction_name}      | {conf_str}     | {timestamp}  |", end="\r", flush=True)
            
            # Update screen with direction
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 36)
            text = font.render(f"Direction: {direction_name} ({conf_str})", True, (255, 255, 255))
            screen.blit(text, (50, 120))
            pygame.display.flip()
            
            # Short sleep to avoid overwhelming the CPU
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nDetection stream ended by user.")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()