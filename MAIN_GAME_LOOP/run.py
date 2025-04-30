#GAME LOOP THIS BITCH AH

#Start Game

#Select Song
    #Loading
    #Init Variables
    #DDC (Done separately)
    #Set up Game Board (Done separately)

#Game Start
#running from gui:
import os
import sys
import argparse

# Set python search paths
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../graphics')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DDC')))

# # Run DDC
# import ddc_downloader
# ddc_downloader.main()

# # Run Game GUI
# import ddr_gui
# ddr_gui.DDRGame().run()
# # game.run()




#Game Loop (Main Loop) (Most important loop need to figure out concurrency)
    #NOTES: Make sure all cameras are in sync
    #Update Game board (One thread)
    #Capture Frame (Figure out multi-thread to single pass into model)
    #Pass to Model (Dont wait just keep going)
    #Check Game time (If time end - Go to end game)
    #Back to Capture Frame
    #Whenever Model finishes determining score pass to score increment

#End Game
#End Credits Screen + Display Score
#Allow user to play again or end game
#If end shut off script, must restart script in order to run game again

# Set python search paths
def setup_paths():
    """Set up Python search paths for imports"""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../graphics')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DDC')))

def main():
    """Main function to control game flow"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DDR Game')
    parser.add_argument('--skip-ddc', action='store_true', 
                       help='Skip DDC downloader and go directly to the game')
    args = parser.parse_args()
    
    # Set up paths for imports
    setup_paths()
    
    # Determine if we should skip DDC
    skip_ddc = args.skip_ddc
    if not skip_ddc:
        print("=== DDR Game Launcher ===")
        response = input("Do you want to download a new song with DDC? (y/n): ").strip().lower()
        skip_ddc = response != 'n'
    
    # Run DDC downloader if needed
    if not skip_ddc:
        print("Skipping DDC downloader...")
    else:
        print("Running DDC Downloader...")
        try:
            import ddc_downloader
            try:
                ddc_downloader.main()
            except SystemExit:
                print("DDC Downloader exited.")
            except Exception as e:
                print(f"Error in DDC Downloader: {e}")
                choice = input("Continue to game anyway? (y/n): ").strip().lower()
                if choice != 'y':
                    print("Exiting...")
                    return
        except ImportError:
            print("Warning: DDC Downloader module not found. Continuing to game...")
    
    # Start the game
    print("Starting DDR Game...")
    try:
        import ddr_gui
        ddr_gui.DDRGame().run()
    except ImportError:
        print("Error: Could not import DDR GUI module. Make sure it's installed.")
        return

if __name__ == "__main__":
    main()