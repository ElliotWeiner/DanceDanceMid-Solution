import os
import time
import subprocess
import zipfile
import shutil
import yt_dlp
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def sanitize_difficulty(input_diff):
    valid_difficulties = ["Beginner", "Easy", "Medium", "Hard", "Challenge"]
    input_diff = input_diff.strip().capitalize()

    for diff in valid_difficulties:
        if input_diff.lower() == diff.lower():
            return diff

    print(f"Invalid difficulty '{input_diff}'. Defaulting to 'Medium'.")
    return "Medium"

def download_audio(youtube_url):
    print("Downloading audio from YouTube...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloaded_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    print("Audio download complete.")

    # Check file size
    mp3_file = 'downloaded_audio.mp3'
    size_MB = os.path.getsize(mp3_file) / (1024 * 1024)
    print(f"Downloaded MP3 size: {size_MB:.2f} MB")

    if size_MB > 16:
        print("MP3 is too large. Compressing to smaller bitrate...")

        compressed_file = 'compressed_audio.mp3'
        subprocess.run([
            'ffmpeg', '-y', '-i', mp3_file, '-b:a', '128k', compressed_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        os.remove(mp3_file)
        os.rename(compressed_file, mp3_file)

        size_MB_after = os.path.getsize(mp3_file) / (1024 * 1024)
        print(f"Compressed MP3 size: {size_MB_after:.2f} MB")

    return mp3_file

def setup_driver(download_folder):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920x1080')
    options.add_experimental_option('prefs', {
        "profile.default_content_settings.popups": 0,
        "download.default_directory": download_folder,
        "directory_upgrade": True
    })

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Allow downloads in headless mode
    driver.execute_cdp_cmd(
        "Page.setDownloadBehavior",
        {
            "behavior": "allow",
            "downloadPath": download_folder,
        }
    )

    return driver

def wait_for_zip_file(download_folder, timeout=180):
    print("Waiting for the new zip file to download...")

    existing_zips = set(os.listdir(download_folder))

    end_time = time.time() + timeout
    new_zip_path = None

    while time.time() < end_time:
        current_files = set(os.listdir(download_folder))
        new_files = current_files - existing_zips
        zip_files = [f for f in new_files if f.endswith('.zip')]

        if zip_files:
            new_zip_path = os.path.join(download_folder, zip_files[0])
            print(f"New download complete: {new_zip_path}")
            return new_zip_path

        time.sleep(2)

    print("Timeout reached, no new zip file detected.")
    return None

def upload_fill_and_submit(driver, file_path, artist_name, song_title, difficulty="Easy"):
    print("Reloading Dance Dance Convolution website...")
    driver.get("https://ddc.chrisdonahue.com/")
    time.sleep(2)  # Let the page load

    wait = WebDriverWait(driver, 30)

    print("Uploading MP3 file...")
    upload_input = wait.until(EC.presence_of_element_located((By.XPATH, '//input[@type="file"]')))
    upload_input.send_keys(os.path.abspath(file_path))
    time.sleep(2)

    print("Filling in artist and title...")
    inputs = driver.find_elements(By.XPATH, '//input[@type="text"]')
    artist_input = inputs[0]
    title_input = inputs[1]

    artist_input.send_keys(artist_name)
    title_input.send_keys(song_title)
    time.sleep(1)

    print(f"Selecting difficulty: {difficulty}")
    difficulty_radio = driver.find_element(By.XPATH, f'//input[@type="radio" and @name="diff_coarse" and @value="{difficulty}"]')
    difficulty_radio.click()
    time.sleep(1)

    print("Waiting for Submit button to be clickable...")
    submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//input[@id="choreograph-submit"]')))

    print("Clicking Submit button now!")
    submit_button.click()

    print("Submitted! Now waiting for file to appear...")
    return wait_for_zip_file(os.path.join(os.getcwd(), 'downloads'))

def process_downloaded_zip(zip_path, artist_name, song_title, difficulty):
    print("Processing downloaded ZIP...")

    temp_extract_folder = os.path.join(os.getcwd(), "temp_extracted")
    if not os.path.exists(temp_extract_folder):
        os.makedirs(temp_extract_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_folder)

    # Find the .sm file
    sm_file_path = None
    for root, dirs, files in os.walk(temp_extract_folder):
        for file in files:
            if file.endswith(".sm"):
                sm_file_path = os.path.join(root, file)
                break

    if not sm_file_path:
        print("No .sm file found inside the ZIP!")
        return

    # Create Step Board Files folder
    step_board_folder = os.path.join(os.getcwd(), "Step Board Files")
    if not os.path.exists(step_board_folder):
        os.makedirs(step_board_folder)

    # Create new filename
    safe_song_title = song_title.replace("/", "_")
    safe_artist_name = artist_name.replace("/", "_")
    new_filename = f"{safe_song_title} by {safe_artist_name} - Difficulty: {difficulty}.sm"
    new_filepath = os.path.join(step_board_folder, new_filename)

    shutil.move(sm_file_path, new_filepath)

    print(f"Saved cleaned step file: {new_filepath}")

    # Cleanup
    shutil.rmtree(temp_extract_folder)

def ddc():
    download_folder = os.path.join(os.getcwd(), 'downloads')
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    youtube_url = input("Enter YouTube URL: ").strip()
    artist_name = input("Enter Artist Name: ").strip()
    song_title = input("Enter Song Title: ").strip()
    difficulty_input = input("Select Difficulty (Beginner, Easy, Medium, Hard, Challenge): ").strip()

    difficulty = sanitize_difficulty(difficulty_input)

    mp3_file = download_audio(youtube_url)

    driver = setup_driver(download_folder)

    try:
        zip_path = upload_fill_and_submit(driver, mp3_file, artist_name, song_title, difficulty)
    finally:
        driver.quit()

    if os.path.exists(mp3_file):
        os.remove(mp3_file)
        print("Temporary MP3 file deleted.")

    if zip_path:
        process_downloaded_zip(zip_path, artist_name, song_title, difficulty)

    print("Done! All files processed.")

if __name__ == "__main__":
    ddc()
