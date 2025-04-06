import os
import re
import pygame
import numpy
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ARROW_SIZE = 64
RECEPTOR_Y = 100
SCROLL_SPEED = 500  # pixels per second
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (200, 0, 200)
CYAN = (0, 255, 255)

@dataclass
class Note:
    direction: int  # 0=left, 1=down, 2=up, 3=right
    beat_time: float
    hold_end_time: Optional[float] = None
    is_hold: bool = False
    is_mine: bool = False
    is_hit: bool = False
    y_pos: float = 0

@dataclass
class SMData:
    title: str = ""
    artist: str = ""
    bpm: float = 120.0
    offset: float = 0.0
    notes: List[Note] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []

class DDRGame:
    def __init__(self):
        # Initialize pygame after setup
        self.pygame_initialized = False
        self.initialize_pygame()
        self.setup_game()
        
    def initialize_pygame(self):
        """Initialize or reinitialize pygame"""
        if self.pygame_initialized:
            pygame.quit()
            
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Python DDR Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)
        self.pygame_initialized = True
        
        self.load_assets()
        
    def load_assets(self):
        """Load images and sounds"""
        self.arrow_images = []
        self.receptor_images = []
        
        # Create arrow surfaces
        for color in [BLUE, RED, GREEN, YELLOW]:  # Left, Down, Up, Right
            # Regular arrow
            arrow_surface = pygame.Surface((ARROW_SIZE, ARROW_SIZE), pygame.SRCALPHA)
            pygame.draw.polygon(arrow_surface, color, [
                (ARROW_SIZE//2, 0),
                (ARROW_SIZE, ARROW_SIZE//2),
                (3*ARROW_SIZE//4, ARROW_SIZE//2),
                (3*ARROW_SIZE//4, ARROW_SIZE),
                (ARROW_SIZE//4, ARROW_SIZE),
                (ARROW_SIZE//4, ARROW_SIZE//2),
                (0, ARROW_SIZE//2)
            ])
            self.arrow_images.append(arrow_surface)
            
            # Receptor (stationary arrow)
            receptor_surface = pygame.Surface((ARROW_SIZE, ARROW_SIZE), pygame.SRCALPHA)
            pygame.draw.polygon(receptor_surface, (*color, 128), [
                (ARROW_SIZE//2, 0),
                (ARROW_SIZE, ARROW_SIZE//2),
                (3*ARROW_SIZE//4, ARROW_SIZE//2),
                (3*ARROW_SIZE//4, ARROW_SIZE),
                (ARROW_SIZE//4, ARROW_SIZE),
                (ARROW_SIZE//4, ARROW_SIZE//2),
                (0, ARROW_SIZE//2)
            ])
            pygame.draw.polygon(receptor_surface, (*color, 200), [
                (ARROW_SIZE//2, 5),
                (ARROW_SIZE-5, ARROW_SIZE//2),
                (3*ARROW_SIZE//4-5, ARROW_SIZE//2),
                (3*ARROW_SIZE//4-5, ARROW_SIZE-5),
                (ARROW_SIZE//4+5, ARROW_SIZE-5),
                (ARROW_SIZE//4+5, ARROW_SIZE//2),
                (5, ARROW_SIZE//2)
            ], 3)
            self.receptor_images.append(receptor_surface)
            
        # Rotate images for different directions
        self.arrow_images[0] = pygame.transform.rotate(self.arrow_images[0], 90)  # Left
        self.arrow_images[1] = pygame.transform.rotate(self.arrow_images[1], 0)   # Down
        self.arrow_images[2] = pygame.transform.rotate(self.arrow_images[2], 180) # Up
        self.arrow_images[3] = pygame.transform.rotate(self.arrow_images[3], 270) # Right
        
        self.receptor_images[0] = pygame.transform.rotate(self.receptor_images[0], 90)  # Left
        self.receptor_images[1] = pygame.transform.rotate(self.receptor_images[1], 0)   # Down
        self.receptor_images[2] = pygame.transform.rotate(self.receptor_images[2], 180) # Up
        self.receptor_images[3] = pygame.transform.rotate(self.receptor_images[3], 270) # Right
        
        # Create simple sound objects (fixed to avoid sndarray issue)
        try:
            # Simple sound initialization
            self.hit_sound = pygame.mixer.Sound(buffer=bytes([128] * 1024))
            self.miss_sound = pygame.mixer.Sound(buffer=bytes([128] * 1024))
        except Exception as e:
            print(f"Error creating sound: {e}")
            # Create null sounds as a fallback
            self.hit_sound = pygame.mixer.Sound(buffer=bytes([0]))
            self.miss_sound = pygame.mixer.Sound(buffer=bytes([0]))

    def setup_game(self):
        """Initialize game variables"""
        self.sm_data = None
        self.current_song = None
        self.game_state = "menu"  # menu, playing, results, file_selector, message
        self.song_time = 0
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.hits = 0
        self.misses = 0
        self.judgments = []  # List of tuples (time, text, position)
        self.key_states = [False, False, False, False]  # Left, Down, Up, Right
        self.file_selector_state = {}
        self.message = ""
        
    def select_sm_file(self):
        """Show a pygame-based file selector for .sm files"""
        # Get the current directory
        songs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "songs")
        
        # Create songs directory if it doesn't exist
        if not os.path.exists(songs_dir):
            os.makedirs(songs_dir)
            print(f"Created songs directory at {songs_dir}")
            print("Please place your StepMania (.sm) files in this directory and restart the game.")
            return
            
        # Get list of .sm files
        sm_files = []
        for root, dirs, files in os.walk(songs_dir):
            for file in files:
                if file.endswith(".sm"):
                    sm_files.append(os.path.join(root, file))
        
        if not sm_files:
            # No .sm files found, show a message
            self.game_state = "message"
            self.message = f"No .sm files found in {songs_dir}.\nPlease add some and restart."
            print(f"No .sm files found in {songs_dir}")
            return
            
        # File selector state
        self.file_selector_state = {
            "files": sm_files,
            "selected_index": 0,
            "scroll_offset": 0,
            "max_display": 10  # Maximum number of files to display at once
        }
        
        self.game_state = "file_selector"
        
    def handle_file_selection(self):
        """Handle file selection screen events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.game_state = "menu"
                    return True
                    
                elif event.key == pygame.K_UP:
                    self.file_selector_state["selected_index"] = max(
                        0, self.file_selector_state["selected_index"] - 1)
                    
                    # Adjust scroll if necessary
                    if (self.file_selector_state["selected_index"] < 
                        self.file_selector_state["scroll_offset"]):
                        self.file_selector_state["scroll_offset"] = self.file_selector_state["selected_index"]
                        
                elif event.key == pygame.K_DOWN:
                    self.file_selector_state["selected_index"] = min(
                        len(self.file_selector_state["files"]) - 1, 
                        self.file_selector_state["selected_index"] + 1)
                    
                    # Adjust scroll if necessary
                    if (self.file_selector_state["selected_index"] >= 
                        self.file_selector_state["scroll_offset"] + self.file_selector_state["max_display"]):
                        self.file_selector_state["scroll_offset"] = (self.file_selector_state["selected_index"] - 
                                                                  self.file_selector_state["max_display"] + 1)
                        
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    # Load the selected file
                    file_path = self.file_selector_state["files"][self.file_selector_state["selected_index"]]
                    self.load_song(file_path)
                    
        return True
        
    def load_song(self, file_path):
        """Load a song from a .sm file path"""
        self.load_sm_file(file_path)
        self.game_state = "playing"
        self.song_time = 0
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.hits = 0
        self.misses = 0
        self.judgments = []
        
        # Try to load associated music file
        music_file = None
        song_dir = os.path.dirname(file_path)
        
        # Look for the music file referenced in the SM file
        if self.sm_data and self.sm_data.title:
            # Try common extensions
            for ext in ['.mp3', '.ogg', '.wav']:
                possible_file = os.path.join(song_dir, self.sm_data.title + ext)
                if os.path.exists(possible_file):
                    music_file = possible_file
                    break
        
        if music_file:
            try:
                pygame.mixer.music.load(music_file)
                pygame.mixer.music.play()
                self.current_song = music_file
            except Exception as e:
                print(f"Error loading music file: {e}")
                self.current_song = None
        
    def load_sm_file(self, file_path):
        """Parse and load SM file data"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sm_content = f.read()
            
            # Create a new SMData object
            self.sm_data = SMData()
            
            # Extract basic metadata
            title_match = re.search(r'#TITLE:(.*?);', sm_content)
            if title_match:
                self.sm_data.title = title_match.group(1).strip()
                
            artist_match = re.search(r'#ARTIST:(.*?);', sm_content)
            if artist_match:
                self.sm_data.artist = artist_match.group(1).strip()
                
            offset_match = re.search(r'#OFFSET:(.*?);', sm_content)
            if offset_match:
                try:
                    self.sm_data.offset = float(offset_match.group(1).strip())
                except ValueError:
                    self.sm_data.offset = 0.0
                    
            # Extract BPM
            bpm_match = re.search(r'#BPMS:(.*?);', sm_content, re.DOTALL)
            if bpm_match:
                bpm_data = bpm_match.group(1).strip()
                # In case of multiple BPMs, just use the first one
                first_bpm = re.search(r'0.000=(\d+\.\d+)', bpm_data)
                if first_bpm:
                    self.sm_data.bpm = float(first_bpm.group(1))
                    
            # Extract notes
            # Find the first chart (usually the easiest one)
            notes_match = re.search(r'#NOTES:[^:]*:[^:]*:[^:]*:[^:]*:[^:]*:(.*?)(?=;)', sm_content, re.DOTALL)
            if notes_match:
                self.parse_notes(notes_match.group(1))
                
        except Exception as e:
            print(f"Error loading SM file: {e}")
            self.sm_data = None
            
    def parse_notes(self, notes_content):
        """Parse the notes section of the SM file"""
        # Clean up the notes content
        notes_content = notes_content.strip()
        
        # Split the notes by measures (separated by commas)
        measures = notes_content.split(',')
        
        # Process each measure
        beat_time = 0.0
        measure_index = 0
        
        for measure in measures:
            # Clean up the measure
            rows = [row.strip() for row in measure.strip().split('\n') if row.strip()]
            num_rows = len(rows)
            
            if num_rows > 0:
                # Calculate beat duration for this measure
                beat_duration = 4.0 / num_rows  # 4 beats per measure divided by number of rows
                
                # Process each row in the measure
                for row_index, row in enumerate(rows):
                    # If row is not the right length (should be 4 for single mode), skip it
                    if len(row) != 4:
                        continue
                    
                    # Calculate the beat time for this row
                    current_beat_time = beat_time + (row_index * beat_duration)
                    
                    # Check each column for notes
                    for col_index, note_char in enumerate(row):
                        if note_char == '1':  # Regular note
                            self.sm_data.notes.append(Note(
                                direction=col_index,
                                beat_time=current_beat_time
                            ))
                        elif note_char == '2':  # Hold head
                            # Look for the end of the hold (3) in the same column
                            hold_end_time = None
                            for search_row in range(row_index + 1, num_rows):
                                if rows[search_row][col_index] == '3':
                                    hold_end_time = beat_time + (search_row * beat_duration)
                                    break
                            
                            self.sm_data.notes.append(Note(
                                direction=col_index,
                                beat_time=current_beat_time,
                                hold_end_time=hold_end_time,
                                is_hold=True
                            ))
                            
                        elif note_char == 'M':  # Mine
                            self.sm_data.notes.append(Note(
                                direction=col_index,
                                beat_time=current_beat_time,
                                is_mine=True
                            ))
                
                # Update the beat time for the next measure
                beat_time += 4.0  # 4 beats per measure
                measure_index += 1
        
        # Sort notes by beat time
        self.sm_data.notes.sort(key=lambda x: x.beat_time)
    
    def beat_time_to_seconds(self, beat_time):
        """Convert beat time to seconds based on BPM"""
        return beat_time * (60.0 / self.sm_data.bpm) - self.sm_data.offset
    
    def handle_events(self):
        """Process game events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if self.game_state == "menu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.select_sm_file()
                    elif event.key == pygame.K_ESCAPE:
                        return False
                        
            elif self.game_state == "file_selector":
                return self.handle_file_selection()
                
            elif self.game_state == "message":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        self.game_state = "menu"
                        
            elif self.game_state == "playing":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.game_state = "menu"
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop()
                    elif event.key == pygame.K_LEFT:
                        self.key_states[0] = True
                        self.check_note_hit(0)
                    elif event.key == pygame.K_DOWN:
                        self.key_states[1] = True
                        self.check_note_hit(1)
                    elif event.key == pygame.K_UP:
                        self.key_states[2] = True
                        self.check_note_hit(2)
                    elif event.key == pygame.K_RIGHT:
                        self.key_states[3] = True
                        self.check_note_hit(3)
                        
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        self.key_states[0] = False
                    elif event.key == pygame.K_DOWN:
                        self.key_states[1] = False
                    elif event.key == pygame.K_UP:
                        self.key_states[2] = False
                    elif event.key == pygame.K_RIGHT:
                        self.key_states[3] = False
                        
            elif self.game_state == "results":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        self.game_state = "menu"
        
        return True
    
    def check_note_hit(self, direction):
        """Check if a note was hit and update score"""
        if not self.sm_data:
            return
            
        # Timing windows
        perfect_window = 0.05  # 50 ms
        great_window = 0.10   # 100 ms
        good_window = 0.15    # 150 ms
        
        # Find the closest note in the given direction
        closest_note = None
        closest_time_diff = float('inf')
        closest_note_index = -1
        
        for i, note in enumerate(self.sm_data.notes):
            if note.direction == direction and not note.is_hit:
                time_diff = abs(self.beat_time_to_seconds(note.beat_time) - self.song_time)
                
                if time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    closest_note = note
                    closest_note_index = i
        
        if closest_note and closest_time_diff < good_window:
            # Determine judgment
            judgment = ""
            points = 0
            
            if closest_time_diff <= perfect_window:
                judgment = "PERFECT"
                points = 100
            elif closest_time_diff <= great_window:
                judgment = "GREAT"
                points = 80
            elif closest_time_diff <= good_window:
                judgment = "GOOD"
                points = 50
                
            # Mark the note as hit
            self.sm_data.notes[closest_note_index].is_hit = True
            
            # Update score and combo
            self.score += points
            self.combo += 1
            self.hits += 1
            self.max_combo = max(self.max_combo, self.combo)
            
            # Add judgment text
            self.judgments.append((self.song_time, judgment, 
                                  (SCREEN_WIDTH // 2, RECEPTOR_Y + 50)))
            
            # Play hit sound
            self.hit_sound.play()
    
    def update(self, dt):
        """Update game state"""
        if self.game_state == "playing":
            # Update song time
            self.song_time += dt
            
            # Check for misses
            if self.sm_data:
                for note in self.sm_data.notes:
                    if not note.is_hit and self.beat_time_to_seconds(note.beat_time) < self.song_time - 0.15:
                        note.is_hit = True
                        self.misses += 1
                        self.combo = 0
                        self.judgments.append((self.song_time, "MISS", 
                                             (SCREEN_WIDTH // 2, RECEPTOR_Y + 50)))
                        self.miss_sound.play()
                
                # Calculate positions for all notes
                for note in self.sm_data.notes:
                    note_time = self.beat_time_to_seconds(note.beat_time)
                    time_diff = note_time - self.song_time
                    note.y_pos = RECEPTOR_Y + time_diff * SCROLL_SPEED
                
                # Check if all notes are hit
                all_notes_hit = all(note.is_hit for note in self.sm_data.notes)
                music_playing = pygame.mixer.music.get_busy() if self.current_song else False
                
                if all_notes_hit and not music_playing:
                    self.game_state = "results"
            
            # Clean up old judgments
            self.judgments = [(t, j, p) for t, j, p in self.judgments 
                             if self.song_time - t < 1.0]
    
    def draw_menu(self):
        """Draw the menu screen"""
        self.screen.fill(BLACK)
        
        title_text = self.font.render("Python DDR Game", True, WHITE)
        instruction_text = self.font.render("Press SPACE to select a .sm file", True, WHITE)
        exit_text = self.font.render("Press ESC to exit", True, WHITE)
        
        self.screen.blit(title_text, 
                         (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 200))
        self.screen.blit(instruction_text, 
                         (SCREEN_WIDTH // 2 - instruction_text.get_width() // 2, 250))
        self.screen.blit(exit_text, 
                         (SCREEN_WIDTH // 2 - exit_text.get_width() // 2, 300))
        
    def draw_game(self):
        """Draw the game screen"""
        self.screen.fill(BLACK)
        
        # Draw receptors
        receptor_x_positions = [SCREEN_WIDTH // 2 - 2*ARROW_SIZE + i*ARROW_SIZE for i in range(4)]
        
        for i in range(4):
            receptor_x = receptor_x_positions[i]
            # Draw pressed state
            if self.key_states[i]:
                glow_surface = pygame.Surface((ARROW_SIZE + 10, ARROW_SIZE + 10), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (255, 255, 255, 128), 
                                 (ARROW_SIZE // 2 + 5, ARROW_SIZE // 2 + 5), ARROW_SIZE // 2 + 5)
                self.screen.blit(glow_surface, (receptor_x - 5, RECEPTOR_Y - 5))
            
            # Draw receptor
            self.screen.blit(self.receptor_images[i], (receptor_x, RECEPTOR_Y))
        
        # Draw notes
        if self.sm_data:
            for note in self.sm_data.notes:
                if not note.is_hit and note.y_pos > 0 and note.y_pos < SCREEN_HEIGHT:
                    note_x = receptor_x_positions[note.direction]
                    self.screen.blit(self.arrow_images[note.direction], (note_x, note.y_pos))
        
        # Draw judgments
        for time, judgment, pos in self.judgments:
            alpha = 255 - int(255 * (self.song_time - time) / 1.0)
            color = WHITE
            if judgment == "PERFECT":
                color = CYAN
            elif judgment == "GREAT":
                color = GREEN
            elif judgment == "GOOD":
                color = YELLOW
            elif judgment == "MISS":
                color = RED
                
            judgment_text = self.font.render(judgment, True, color)
            judgment_text.set_alpha(alpha)
            self.screen.blit(judgment_text, 
                           (pos[0] - judgment_text.get_width() // 2, pos[1]))
        
        # Draw score and combo
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        combo_text = self.font.render(f"Combo: {self.combo}", True, WHITE)
        
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(combo_text, (20, 50))
        
        # Draw song info
        if self.sm_data:
            title_text = self.small_font.render(
                f"{self.sm_data.title} - {self.sm_data.artist}", True, WHITE)
            self.screen.blit(title_text, (SCREEN_WIDTH - title_text.get_width() - 20, 20))
    
    def draw_results(self):
        """Draw the results screen"""
        self.screen.fill(BLACK)
        
        title_text = self.font.render("Results", True, WHITE)
        score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
        combo_text = self.font.render(f"Max Combo: {self.max_combo}", True, WHITE)
        accuracy_text = self.font.render(
            f"Accuracy: {int(self.hits / (self.hits + self.misses) * 100) if (self.hits + self.misses) > 0 else 0}%", 
            True, WHITE)
        continue_text = self.font.render("Press SPACE to continue", True, WHITE)
        
        self.screen.blit(title_text, 
                         (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 150))
        self.screen.blit(score_text, 
                         (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 200))
        self.screen.blit(combo_text, 
                         (SCREEN_WIDTH // 2 - combo_text.get_width() // 2, 250))
        self.screen.blit(accuracy_text, 
                         (SCREEN_WIDTH // 2 - accuracy_text.get_width() // 2, 300))
        self.screen.blit(continue_text, 
                         (SCREEN_WIDTH // 2 - continue_text.get_width() // 2, 400))
    
    def draw_file_selector(self):
        """Draw the file selector screen"""
        self.screen.fill(BLACK)
        
        title_text = self.font.render("Select a StepMania File", True, WHITE)
        self.screen.blit(title_text, 
                       (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 50))
        
        instruction_text = self.small_font.render("UP/DOWN to navigate, ENTER to select, ESC to cancel", True, WHITE)
        self.screen.blit(instruction_text, 
                       (SCREEN_WIDTH // 2 - instruction_text.get_width() // 2, 80))
        
        # Display files
        y_position = 120
        display_start = self.file_selector_state["scroll_offset"]
        display_end = min(display_start + self.file_selector_state["max_display"],
                         len(self.file_selector_state["files"]))
        
        for i in range(display_start, display_end):
            file_path = self.file_selector_state["files"][i]
            file_name = os.path.basename(file_path)
            
            # Highlight selected file
            if i == self.file_selector_state["selected_index"]:
                pygame.draw.rect(self.screen, BLUE, 
                               (100, y_position, SCREEN_WIDTH - 200, 30))
                text_color = WHITE
            else:
                text_color = GRAY
                
            file_text = self.small_font.render(file_name, True, text_color)
            self.screen.blit(file_text, (120, y_position + 5))
            
            y_position += 40
            
        # Draw scroll indicators if needed
        if display_start > 0:
            up_text = self.font.render("▲", True, WHITE)
            self.screen.blit(up_text, 
                           (SCREEN_WIDTH // 2, 100))
            
        if display_end < len(self.file_selector_state["files"]):
            down_text = self.font.render("▼", True, WHITE)
            self.screen.blit(down_text, 
                           (SCREEN_WIDTH // 2, 500))
            
    def draw_message(self):
        """Draw a message screen"""
        self.screen.fill(BLACK)
        
        lines = self.message.split('\n')
        y_position = SCREEN_HEIGHT // 2 - (len(lines) * 30) // 2
        
        for line in lines:
            text = self.font.render(line, True, WHITE)
            self.screen.blit(text, 
                           (SCREEN_WIDTH // 2 - text.get_width() // 2, y_position))
            y_position += 40
            
        continue_text = self.small_font.render("Press any key to continue", True, GRAY)
        self.screen.blit(continue_text, 
                       (SCREEN_WIDTH // 2 - continue_text.get_width() // 2, 
                        SCREEN_HEIGHT - 80))
    
    def draw(self):
        """Draw the current game state"""
        if self.game_state == "menu":
            self.draw_menu()
        elif self.game_state == "file_selector":
            self.draw_file_selector()
        elif self.game_state == "message":
            self.draw_message()
        elif self.game_state == "playing":
            self.draw_game()
        elif self.game_state == "results":
            self.draw_results()
            
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time in seconds
            running = self.handle_events()
            
            self.update(dt)
            self.draw()
        
        pygame.quit()

if __name__ == "__main__":
    game = DDRGame()
    game.run()