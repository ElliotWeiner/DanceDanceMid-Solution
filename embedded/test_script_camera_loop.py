import os
import sys
import threading
import subprocess
import io
import queue
import time
import socket
import json
import numpy as np
import torch
import torchvision.transforms as T
from datetime import datetime
from PIL import Image, ImageSequence

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RTSP_URLS = {
    2: "rtsp://root:botbot@192.168.1.131/axis-media/media.amp",
    3: "rtsp://root:botbot@192.168.1.160/axis-media/media.amp",
}

# point this at your ffmpeg bin if necessary
# os.environ["PATH"] = r"C:\path\to\ffmpeg\bin" + os.pathsep + os.environ["PATH"]

OUTPUT_ROOT      = os.path.abspath("Model Frames")
BUFFER_LEN       = 3
NUM_INFER_WORKERS = 2

# ─── SOCKET SERVER ─────────────────────────────────────────────────────────────
class SocketServer:
    def __init__(self, host="localhost", port=12345):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket = None
        self.counter = 0
        self.running = True

    def start(self):
        t = threading.Thread(target=self._run_server, daemon=True)
        t.start()

    def _run_server(self):
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"[Socket] Listening on {self.host}:{self.port}")
            while self.running:
                client, addr = self.server_socket.accept()
                self.client_socket = client
                print(f"[Socket] Client connected: {addr}")
                # hold the connection open
                while self.running and self.client_socket:
                    time.sleep(0.1)
        except Exception as e:
            print(f"[Socket] Error: {e}")
        finally:
            self.close()

    def send_direction(self, direction_code: int):
        if not self.client_socket:
            return
        directions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "NONE"}
        direction_name = directions.get(direction_code, "UNKNOWN")
        self.counter += 1
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        msg = {
            "counter": self.counter,
            "direction_code": direction_code,
            "direction": direction_name,
            "timestamp": ts,
            "message": direction_name
        }
        try:
            self.client_socket.sendall(json.dumps(msg).encode("utf-8") + b"\n")
            print(f"[Socket] Sent {direction_name} @ {ts}", end="\r", flush=True)
        except:
            print("\n[Socket] Connection lost.")
            self.client_socket.close()
            self.client_socket = None

    def close(self):
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        self.server_socket.close()
        print("\n[Socket] Server closed")


socket_server = SocketServer()

# ─── PIPELINE QUEUES & STATE ───────────────────────────────────────────────────
frame_queue     = queue.Queue(maxsize=500)   # (cam_id, [PIL imgs], ts)
inference_queue = queue.Queue(maxsize=100)   # {cam_id: gif_path, ...}
merge_buffer    = {}
merge_lock      = threading.Lock()
stop_event      = threading.Event()

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def start_ffmpeg_stream(rtsp_url):
    return subprocess.Popen([
        "ffmpeg", "-rtsp_transport", "tcp", "-i", rtsp_url,
        "-loglevel", "quiet", "-vf", "fps=10", "-q:v", "5",
        "-f", "mjpeg", "-"
    ], stdout=subprocess.PIPE, bufsize=0)

# your test-time transform
transform_test = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize((0.4316, 0.3945, 0.3765), (0.2280, 0.2215, 0.2170)),
])

# placeholder model
def run_model(cam1_tensor, cam2_tensor):
    # TODO: replace with your actual model call
    # e.g. logits = model(cam1_tensor.unsqueeze(0), cam2_tensor.unsqueeze(0))
    # direction_code = logits.argmax(dim=1).item()
    return 0  # placeholder UP

# ─── THREADS ───────────────────────────────────────────────────────────────────
def camera_worker(cam_id, rtsp_url):
    while not stop_event.is_set():
        proc   = start_ffmpeg_stream(rtsp_url)
        buffer = bytearray()
        frames = []
        print(f"[Camera{cam_id}] FFmpeg started.")
        try:
            while not stop_event.is_set():
                chunk = proc.stdout.read(4096)
                if not chunk:
                    raise IOError("EOF from FFmpeg")
                buffer.extend(chunk)
                # extract complete JPEGs
                while True:
                    soi = buffer.find(b'\xff\xd8')
                    eoi = buffer.find(b'\xff\xd9', soi+2)
                    if soi<0 or eoi<0: break
                    jpeg = bytes(buffer[soi:eoi+2]); del buffer[:eoi+2]
                    try:
                        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                    except:
                        continue
                    frames.append(img)
                    print(f"[Camera{cam_id}] frame {len(frames)}/{BUFFER_LEN}")
                    if len(frames)==BUFFER_LEN:
                        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                        try:
                            frame_queue.put_nowait((cam_id, list(frames), ts))
                            print(f"[Camera{cam_id}] enqueued @ {ts}")
                        except queue.Full:
                            print(f"[Camera{cam_id}] frame_queue FULL")
                        frames.clear()
        except Exception as e:
            print(f"[Camera{cam_id}] Error: {e}  → restarting FFmpeg")
        finally:
            proc.kill()
    print(f"[Camera{cam_id}] exiting.")

def merging_worker():
    while not stop_event.is_set():
        try:
            cam_id, frames, ts = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        with merge_lock:
            merge_buffer[cam_id] = (frames, ts)
            if set(merge_buffer)==set(RTSP_URLS):
                gifs = {}
                for cid,(frm,st) in merge_buffer.items():
                    path = os.path.join(OUTPUT_ROOT, f"cam_{cid}", f"cam{cid}_{st}.gif")
                    frm[0].save(path, save_all=True, append_images=frm[1:], duration=100, loop=0)
                    print(f"[Merger] wrote {path}")
                    gifs[cid]=path
                merge_buffer.clear()
                try:
                    inference_queue.put_nowait(gifs)
                    print(f"[Merger] job→inference {gifs}")
                except queue.Full:
                    print("[Merger] inference_queue FULL")
    print("[Merger] exiting.")

def processing_worker(wid):
    while not stop_event.is_set():
        try:
            gif_paths = inference_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        cam_tensors = {}
        for cid,path in gif_paths.items():
            gif = Image.open(path)
            raw = []
            for i,frame in enumerate(ImageSequence.Iterator(gif)):
                if i>=BUFFER_LEN: break
                raw.append(frame.convert("RGB"))
            gif.close()

            arr  = np.stack([np.array(x) for x in raw],axis=0)                  # (3,H,W,C)
            crop = arr[:,112:480-112,224:704-224,:]                             # your crop
            t    = torch.from_numpy(crop).permute(0,3,2,1)                      # (3,C,W,H)
            t_t  = transform_test(t)                                            # (3,C,112,112)
            cam_tensors[cid]=t_t
            print(f"[Infer{wid}] cam{cid}: {t_t.shape}")

        # run your model
        code = run_model(cam_tensors[2], cam_tensors[3])
        print(f"[Infer{wid}] model→direction {code}")
        socket_server.send_direction(code)
    print(f"[Infer{wid}] exiting.")

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # prepare folders
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for cid in RTSP_URLS:
        os.makedirs(os.path.join(OUTPUT_ROOT, f"cam_{cid}"), exist_ok=True)

    # start socket server
    socket_server.start()

    # start merging & camera threads
    threading.Thread(target=merging_worker, daemon=True).start()
    for cid,url in RTSP_URLS.items():
        threading.Thread(target=camera_worker, args=(cid,url), daemon=True).start()

    # start inference workers
    for i in range(NUM_INFER_WORKERS):
        threading.Thread(target=processing_worker, args=(i,), daemon=True).start()

    print(f"[+] Running. OUTPUT_ROOT={OUTPUT_ROOT}. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[!] Shutting down…")
        stop_event.set()
        time.sleep(1)
        socket_server.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
