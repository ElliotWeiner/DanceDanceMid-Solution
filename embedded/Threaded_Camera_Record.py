#This is the threaded camera recording file. You need to have ffmpeg installed
#Note: This file will save all the .mp4 files whereever the python file is stored.

#If CONDA look up online if PIP install its: pip install thread6
import threading
#Already built into python no need to download
import subprocess
import os
import glob

#Sets file pathway were the folders and files will exist
script_dir = os.path.dirname(os.path.abspath(__file__))


#Defining RTSP URLS (ONLY NEED TO CHECK/CHANGE IF THE CAMERAS LOSE POWER AND TURN OFF)
cameras = {
    1: 'rtsp://root:botbot@192.168.0.111/axis-media/media.amp',
    2: 'rtsp://root:botbot@192.168.0.114/axis-media/media.amp',
    3: 'rtsp://root:botbot@192.168.0.129/axis-media/media.amp',
    4: 'rtsp://root:botbot@192.168.0.134/axis-media/media.amp',
}

#Makes sure camera folder exists
for cam_num in cameras:
    cam_folder = os.path.join(script_dir, f"cam{cam_num}")
    os.makedirs(cam_folder, exist_ok=True)

start_barrier = threading.Barrier(len(cameras))



class CamThread(threading.Thread):
    def __init__(self, ffmpeg_path, cam_num, rtsp_url):
        super().__init__()
        self.ffmpeg = ffmpeg_path
        self.cam_num = cam_num
        self.rtsp = rtsp_url

        # — find next 5‑digit index for this camera —
        cam_folder = os.path.join(script_dir, f"cam{cam_num}")
        pattern = os.path.join(cam_folder, "output_*.mp4")
        existing = glob.glob(pattern)

        nums = []
        for path in existing:
            name = os.path.splitext(os.path.basename(path))[0]
            # name is like "output_00005"
            parts = name.split('_')
            if len(parts) == 2 and parts[0] == "output":
                try:
                    nums.append(int(parts[1]))
                except ValueError:
                    pass

        next_index = (max(nums) + 1) if nums else 1

        # — build zero‑padded 5‑digit filename —
        fname = f"output_{next_index:05d}.mp4"
        self.outfile = os.path.join(cam_folder, fname)

    def run(self):
        print(f"[Camera {self.cam_num}] Recording → {self.outfile}")
        start_barrier.wait()
        self.record()

    def record(self):
        cmd = [    
            self.ffmpeg,
            "-i", self.rtsp,
            "-c", "copy",
            "-t", "00:01:10",  #SET DURATION HERE
            self.outfile
        ]
        #print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode == 0:
            print(f"[Camera {self.cam_num}] Done.")
        else:
            print(f"[Camera {self.cam_num} ERROR]:", err.decode())

def get_frame_count(video_path):
    cmd = [
        ffprobe_exe,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = proc.communicate()
    try:
        return int(out.strip())
    except Exception:
        return None


#Need to download FFMPEG and set path to YOUR SPECIFIC FILE PATHWAY TO THE .EXE file
#Windows ffmpeg download windows > gyan.dev > essentials only
ffmpeg_exe = rpath = r"C:\Users\hummy\Downloads\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\bin\ffmpeg.exe"

ffprobe_exe = r"C:\Users\hummy\Downloads\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\bin\ffprobe.exe"

threads = []

#Starts Cameras and Records
print("Starting Recording.\n")
for num, url in cameras.items():
    t = CamThread(ffmpeg_exe, num, url)
    t.start()
    threads.append(t)

#Joins files to respective folders
for t in threads:
    t.join()

#Tells you its complete
print("All recordings complete.")

for t in threads:
    frames = get_frame_count(t.outfile)
    if frames is not None:
        print(f"{t.outfile}: {frames} frames")
    else:
        print(f"{t.outfile}: Unable to determine frame count")


