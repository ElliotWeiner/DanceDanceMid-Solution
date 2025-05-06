# This is the threaded camera recording file. You need to have ffmpeg installed
# Note: This file will save all the .mp4 files whereever the python file is stored.

import threading
import io
import subprocess
import os
from collections import deque
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# PLEASE CHANGE THIS TO YOUR FILE PATH IF ITS ALREADY SQUARED AWAY THEN DELETE
ffmpeg_dir = r"C:\Users\hummy\Downloads\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\bin"
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


# Remember to comment out the two cameras that are not in use
cameras = {
    # 1: 'rtsp://root:botbot@192.168.0.116/axis-media/media.amp',
    2: "rtsp://root:botbot@192.168.0.115/axis-media/media.amp",
    3: "rtsp://root:botbot@192.168.0.135/axis-media/media.amp",
    # 4: 'rtsp://root:botbot@192.168.0.130/axis-media/media.amp',
}
OUTPUT_ROOT = "Model Frames"

# make per-camera folders
for cam_id in cameras:
    os.makedirs(os.path.join(OUTPUT_ROOT, f"cam_{cam_id}"), exist_ok=True)

# Barrier to sync the 4 threads before they grab their frame
barrier = threading.Barrier(len(cameras))


def model_inference(camera_id: int, frames: list[Image.Image]):
    """
    frames: [oldest, middle, newest] for this camera
    """

    # Returns:
    # - 0: up
    # - 1: down
    # - 2: left
    # - 3: right
    # - 4: no input

    return 0

    # print(f"[Camera {camera_id}] model_inference on {len(frames)} frames")

    # logits = model(frames[camera_pov1], frames[camera_pov2])

    # probs = F.softmax(logits, dim=1)
    # _, pred = torch.argmax(probs, dim=1)

    # return pred.detach.cpu().int()


def grab_frame(rtsp_url: str) -> Image.Image:
    """
    Uses FFmpeg to grab one frame from the RTSP stream
    """
    cmd = [
        "ffmpeg",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    jpeg_bytes, _ = proc.communicate()
    return Image.open(io.BytesIO(jpeg_bytes))


def camera_worker(camera_id: int, rtsp_url: str):
    frame_q = deque(maxlen=3)
    counter = 0
    transform_test = transforms.Compose(
        [
            transforms.Resize(size=(112, 112)),
            transforms.Normalize((0.4316, 0.3945, 0.3765), (0.228, 0.2215, 0.2170)),
        ]
    )
    while True:
        # 1) wait until all cameras are ready
        barrier.wait()

        # 2) grab frame
        try:
            img = grab_frame(rtsp_url)
            arr = np.array(img)
            crop = arr[:, 112 : 480 - 112, 224 : 704 - 224, :]
            tensor = torch.from_numpy(crop)
            tensor = tensor.permute(0, 3, 2, 1)
            resized = transform_test(tensor)
            resized = resized.permute(1, 0, 2, 3)

        except Exception as e:
            print(f"[Camera {camera_id}] Error grabbing frame: {e}")
            continue

        # 3) enqueue and save
        frame_q.append(resized)
        fname = f"frame_{counter:06d}.jpg"
        img.save(os.path.join(OUTPUT_ROOT, f"cam_{camera_id}", fname))
        counter += 1

        # 4) once we have 3 frames, call model
        if len(frame_q) == 3:
            model_inference(camera_id, list(frame_q))


def main():
    threads = []
    for cam_id, url in cameras.items():
        t = threading.Thread(target=camera_worker, args=(cam_id, url), daemon=True)
        t.start()
        threads.append(t)

    print("[+] All camera threads started. Press Ctrl+C to stop.")
    try:
        # keep the main thread alive
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\n[!] Interrupted—exiting.")


if __name__ == "__main__":

    main()
