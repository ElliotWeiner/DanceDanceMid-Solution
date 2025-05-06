import threading
import io
import subprocess
import os
import socket
import json
import time
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

# Barrier to sync the threads before they grab their frame
barrier = threading.Barrier(len(cameras))


# Socket server setup
class SocketServer:
    def __init__(self, host="localhost", port=12345):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket = None
        self.counter = 0
        self.running = True

    def start(self):
        """Start the socket server in a separate thread"""
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def _run_server(self):
        """Internal method to run the server"""
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"Socket server listening on {self.host}:{self.port}")

            while self.running:
                # Accept connection
                self.client_socket, client_address = self.server_socket.accept()
                print(f"Connection established with: {client_address}")

                # Keep connection open until client disconnects or server stops
                while self.running and self.client_socket:
                    time.sleep(0.1)  # Small sleep to prevent CPU hogging

        except Exception as e:
            print(f"Socket server error: {e}")
        finally:
            self.close()

    def send_direction(self, direction_code):
        """Send direction code to connected client"""
        if not self.client_socket:
            return

        # Get direction name
        direction_name = self.get_direction_name(direction_code)

        # Create message
        self.counter += 1
        timestamp = time.strftime("%H:%M:%S.%f")[:-3]
        message = {
            "counter": self.counter,
            "direction_code": direction_code,
            "direction": direction_name,
            "timestamp": timestamp,
            "message": f"{direction_name}",
        }

        # Send message
        try:
            json_message = json.dumps(message) + "\n"
            self.client_socket.sendall(json_message.encode("utf-8"))
            print(
                f"Sent direction: {direction_name} [{timestamp}]", end="\r", flush=True
            )
        except:
            print("Connection lost. Waiting for new connection...")
            self.client_socket.close()
            self.client_socket = None

    def get_direction_name(self, direction_code):
        """Convert direction code to human-readable name"""
        directions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "NONE"}
        return directions.get(direction_code, "UNKNOWN")

    def close(self):
        """Close the socket server"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        self.server_socket.close()
        print("Socket server closed")


# Create global socket server instance
socket_server = SocketServer()


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
    # TODO: Implement your actual model inference here
    # For now, just returning a placeholder direction (UP)
    direction_code = 0

    # Send the direction to any connected clients
    socket_server.send_direction(direction_code)

    return direction_code

    # Your original commented code:
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
    # Start the socket server
    socket_server.start()

    # Start camera threads
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
        socket_server.close()


if __name__ == "__main__":
    main()
