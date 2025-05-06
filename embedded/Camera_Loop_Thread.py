# This is the threaded camera recording file. You need to have ffmpeg installed
# Note: This file will save all the .mp4 files whereever the python file is stored.

import threading
import io
import subprocess
import os
import socket
import json
from collections import deque
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import time

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

# Barrier to sync the threads before they grab their frame
barrier = threading.Barrier(len(cameras))


# Direction mapping
def get_direction_name(direction_code):
    """Convert direction code to human-readable name"""
    directions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "NONE"}
    return directions.get(direction_code, "UNKNOWN")


# Initialize socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "localhost"
port = 12345
client_socket = None
message_counter = 0


# Start socket server in a separate thread
def socket_server():
    global client_socket
    try:
        # Bind socket to address and port
        server_socket.bind((host, port))
        # Listen for incoming connections
        server_socket.listen(1)
        print(f"Publisher listening on {host}:{port}")

        while True:
            # Accept a connection
            client_socket_local, client_address = server_socket.accept()
            client_socket = client_socket_local
            print(f"Connection established with: {client_address}")

            # Keep the connection open until client disconnects
            while True:
                # This just keeps the thread alive
                time.sleep(1)
    except Exception as e:
        print(f"Socket server error: {e}")
    finally:
        if client_socket:
            client_socket.close()
        server_socket.close()


# Function to send direction via socket
def send_direction(direction_code):
    global client_socket, message_counter

    if client_socket is None:
        return

    message_counter += 1
    direction_name = get_direction_name(direction_code)
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]

    # Create JSON message to send
    message = {
        "counter": message_counter,
        "direction_code": direction_code,
        "direction": direction_name,
        "timestamp": timestamp,
        "message": f"{direction_name}",
    }

    # Send message to client
    try:
        json_message = json.dumps(message) + "\n"
        client_socket.sendall(json_message.encode("utf-8"))
        print(f"Sent direction: {direction_name} at {timestamp}")
    except:
        global client_socket  # This line was causing the syntax error
        print("Connection lost. Waiting for new connection...")
        client_socket = None  # This resets the client_socket to None


def model_inference(camera_id: int, frames: list[Image.Image]):
    """
    frames: [oldest, middle, newest] for this camera

    Returns:
    - 0: up
    - 1: down
    - 2: left
    - 3: right
    - 4: no input
    """
    # This is a placeholder. Replace with actual model inference
    # For now, returning 0 as in the original code
    direction_code = 0

    # Send the direction via socket
    send_direction(direction_code)

    return direction_code


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
    # Start socket server thread
    socket_thread = threading.Thread(target=socket_server, daemon=True)
    socket_thread.start()

    # Start camera threads
    camera_threads = []
    for cam_id, url in cameras.items():
        t = threading.Thread(target=camera_worker, args=(cam_id, url), daemon=True)
        t.start()
        camera_threads.append(t)

    print("[+] All camera threads started. Press Ctrl+C to stop.")
    try:
        # keep the main thread alive
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\n[!] Interrupted—exiting.")
    finally:
        # Clean up socket
        server_socket.close()


if __name__ == "__main__":
    main()
