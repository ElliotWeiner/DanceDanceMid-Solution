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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../graphics')))

import ddr_gui  
game = ddr_gui.DDRGame()
game.run()




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
