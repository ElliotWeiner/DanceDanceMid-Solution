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

def sanitize_filename(filename):
    # Replace invalid characters for filenames
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def sanitize_difficulty(input_diff):
    valid_difficulties = ["Beginner", "Easy", "Medium", "Hard", "Challenge"]
    input_diff = input_diff.strip().capitalize()

    for diff in valid_difficulties:
        if input_diff.lower() == diff.lower():
            return diff

    print(f"Invalid difficulty '{input_diff}'. Defaulting to 'Medium'.")
    return "Medium"

def download_audio(youtube_url, output_folder):
    print("Downloading audio from YouTube...")
    mp3_path = os.path.join(output_folder, 'audio.mp3')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_folder, 'downloaded_audio.%(ext)s'),
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

    # Find the downloaded mp3 file
    for file in os.listdir(output_folder):
        if file.endswith('.mp3'):
            mp3_file = os.path.join(output_folder, file)
            break
    else:
        print("Error: MP3 file not found after download")
        return None

    # Check file size
    size_MB = os.path.getsize(mp3_file) / (1024 * 1024)
    print(f"Downloaded MP3 size: {size_MB:.2f} MB")

    if size_MB > 16:
        print("MP3 is too large. Compressing to smaller bitrate...")

        compressed_file = os.path.join(output_folder, 'compressed_audio.mp3')
        subprocess.run([
            'ffmpeg', '-y', '-i', mp3_file, '-b:a', '128k', compressed_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        os.remove(mp3_file)
        os.rename(compressed_file, mp3_file)

        size_MB_after = os.path.getsize(mp3_file) / (1024 * 1024)
        print(f"Compressed MP3 size: {size_MB_after:.2f} MB")

    # Rename to final output name
    final_mp3 = os.path.join(output_folder, 'audio.mp3')
    if mp3_file != final_mp3:
        os.rename(mp3_file, final_mp3)

    return final_mp3

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

def process_downloaded_zip(zip_path, output_folder, song_title, artist_name):
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

    # Create the new filename in the format 'Title - Artist.sm'
    safe_filename = sanitize_filename(f"{song_title} - {artist_name}")
    new_sm_path = os.path.join(output_folder, f"{safe_filename}.sm")
    
    # Move the .sm file to the output folder with the new name
    shutil.copy(sm_file_path, new_sm_path)
    print(f"Saved step file to: {new_sm_path}")

    # Cleanup
    shutil.rmtree(temp_extract_folder)

def main():
    # Create base songs directory if it doesn't exist
    songs_base_dir = os.path.join(os.getcwd(), 'songs')
    if not os.path.exists(songs_base_dir):
        os.makedirs(songs_base_dir)

    # Create downloads folder if it doesn't exist
    download_folder = os.path.join(os.getcwd(), 'downloads')
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    youtube_url = input("Enter YouTube URL: ").strip()
    artist_name = input("Enter Artist Name: ").strip()
    song_title = input("Enter Song Title: ").strip()
    difficulty_input = input("Select Difficulty (Beginner, Easy, Medium, Hard, Challenge): ").strip()

    difficulty = sanitize_difficulty(difficulty_input)
    
    # Create a safe directory name
    safe_song_name = sanitize_filename(f"{song_title} - {artist_name}")
    song_dir = os.path.join(songs_base_dir, safe_song_name)
    
    # Create song directory if it doesn't exist
    if not os.path.exists(song_dir):
        os.makedirs(song_dir)
    
    # Download audio directly to the song directory
    mp3_file = download_audio(youtube_url, song_dir)
    
    if not mp3_file:
        print("Error: Failed to download or process audio file")
        return

    driver = setup_driver(download_folder)

    try:
        zip_path = upload_fill_and_submit(driver, mp3_file, artist_name, song_title, difficulty)
        
        if zip_path:
            process_downloaded_zip(zip_path, song_dir, song_title, artist_name)            
            # Clean up the downloaded zip file
            os.remove(zip_path)
            print(f"Temporary ZIP file deleted.")
            
            print(f"Success! Files saved to: {song_dir}")
            print(f"  - MP3: {os.path.join(song_dir, 'audio.mp3')}")
            print(f"  - Step file: {os.path.join(song_dir, 'stepfile.sm')}")
    finally:
        driver.quit()

    print("Done! All files processed.")

if __name__ == "__main__":
    main()