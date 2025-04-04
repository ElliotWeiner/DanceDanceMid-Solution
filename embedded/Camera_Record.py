import subprocess
import os
# Replace with your actual RTSP URL
path = r"C:\Users\hummy\Downloads\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\bin\ffmpeg.exe"
rtsp_url = "rtsp://root:botbot@192.168.0.114/axis-media/media.amp"
output_file = "output.mp4"

# FFmpeg command
command = [
    path,
    "-i", rtsp_url,          # input stream
    "-c", "copy",            # no re-encoding
    "-t", "00:00:10",        # duration (1 minute)
    output_file
]

# Run FFmpeg
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Optional: wait for it to finish
stdout, stderr = process.communicate()

# Check result
if process.returncode == 0:
    print("Recording complete.")
    print("Saved to:", os.path.abspath(output_file))
else:
    print("Error:", stderr.decode())
