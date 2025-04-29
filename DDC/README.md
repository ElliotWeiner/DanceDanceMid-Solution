Dance Dance Convolution Auto Stepfile Downloader

-This script automatically:--
Downloads a song from YouTube
Uploads it to the Dance Dance Convolution website (https://ddc.chrisdonahue.com/)
Downloads the generated stepchart (.sm file)
Renames the stepchart file
Organizes the stepchart file inside a folder

--Requirements--
Python 3.8 or newer
Google Chrome installed
ffmpeg installed on the system

--Notes--
If the downloaded song is larger than 16 MB, the script automatically compresses it.
The script uses headless Chrome to interact with the website without opening a visible browser window.
Difficulty input is case-insensitive and auto-corrected if there is a small typo.

--Troubleshooting--
Make sure Chrome is installed and up-to-date.
Ensure that ffmpeg is installed and added to your system's PATH.

If any Selenium errors occur, make sure to update the required Python packages:

pip install --upgrade selenium webdriver-manager yt-dlp
If Chrome version updates, webdriver-manager should automatically fetch the correct version of chromedriver.

Created by Anton Garcia Abril Beca
