import threading
import subprocess
import os

# Define class for the camera thread.
class CamThread(threading.Thread):
    def __init__(self, path, previewname, camid, filename):
        threading.Thread.__init__(self)
        self.path = path
        self.previewname = previewname
        self.camid = camid
        self.outputfile = filename

    def run(self):
        print("Starting " + self.previewname)
        # Corrected parameter order: camid, outputfile, path
        recording(self.camid, self.outputfile, self.path)

# Function to preview the camera.
def recording(camid, outputfile, path):
    command = [
        path,
        "-i", camid,      # input stream
        "-c", "copy",     # no re-encoding
        "-t", "00:00:30", # duration (10 seconds)
        outputfile
    ]

    # Run FFmpeg
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Check result
    if process.returncode == 0:
        print("Recording complete.")
        print("Saved to:", os.path.abspath(outputfile))
    else:
        print("Error:", stderr.decode())

def stitching():
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    # Command to stitch the videos together
    stitch_command = [
    path,  # Path to ffmpeg executable
    '-i', 'output1.mp4',
    '-i', 'output2.mp4',
    '-i', 'output3.mp4',
    '-i', 'output4.mp4',
    '-filter_complex', 'xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0',
    '-c:v', 'libx264',
    'combined_output.mp4'
    ]

    process = subprocess.run(stitch_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode == 0:
        print("Stitching complete. Combined video saved as combined_output.mp4")
    else:
        print("Error during stitching:", process.stderr.decode())

print("chk1")

# Create different threads for each video stream, then start them.
path = r"/Users/elliotweiner/Downloads/ffmpeg"
thread1 = CamThread(path, "Camera1", 'rtsp://root:botbot@192.168.0.114/axis-media/media.amp', '/Users/elliotweiner/Desktop/DDR/DanceDanceMid-Solution/dataset/cam1/output1_1.mp4')
thread2 = CamThread(path, "Camera2", 'rtsp://root:botbot@192.168.0.111/axis-media/media.amp', '/Users/elliotweiner/Desktop/DDR/DanceDanceMid-Solution/dataset/cam2/output1_2.mp4')
thread3 = CamThread(path, "Camera3", 'rtsp://root:botbot@192.168.0.129/axis-media/media.amp', '/Users/elliotweiner/Desktop/DDR/DanceDanceMid-Solution/dataset/cam3/output1_3.mp4')
thread4 = CamThread(path, "Camera4", 'rtsp://root:botbot@192.168.0.134/axis-media/media.amp', '/Users/elliotweiner/Desktop/DDR/DanceDanceMid-Solution/dataset/cam4/output1_4.mp4')

print("chk2")

thread1.start()
thread2.start()
thread3.start()
thread4.start()

# IF YOU WANT TO STICH ALL THE VIDEOS TOGETHER UNCOMMENT AND RUN THE CODE BELOW
#stitching()
