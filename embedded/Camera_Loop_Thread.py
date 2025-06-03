import os
import sys
import threading
import subprocess
import io
import queue
import time
import socket
import json
import traceback
import numpy as np
import torch
import torchvision.transforms as T
from FeetNet import FeetNet
from datetime import datetime
from PIL import Image, ImageSequence

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RTSP_URLS = {
    2: "rtsp://root:botbot@192.168.0.108/axis-media/media.amp",
    3: "rtsp://root:botbot@192.168.0.131/axis-media/media.amp",
}
OUTPUT_ROOT = os.path.abspath("Model Frames")
BUFFER_LEN = 3
NUM_INFER_WORKERS = 2


# ─── SOCKET SERVER ────────────────────────────────────────────────────────────
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
        name = directions.get(direction_code, "UNKNOWN")
        self.counter += 1
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        msg = {
            "counter": self.counter,
            "direction_code": direction_code,
            "direction": name,
            "timestamp": ts,
            "message": name,
        }
        try:
            print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
            self.client_socket.sendall(json.dumps(msg).encode() + b"\n")
            print(f"[Socket] Sent {name} @ {ts}", end="\r", flush=True)
        except:
            print("\n[Socket] Connection lost.")
            self.client_socket.close()
            self.client_socket = None

    def close(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        try:
            self.server_socket.close()
        except:
            pass
        print("\n[Socket] Server closed")


socket_server = SocketServer()

# ─── QUEUES & STATE ────────────────────────────────────────────────────────────
frame_queue = queue.Queue(maxsize=500)  # (cam_id, [PIL imgs], ts)
inference_queue = queue.Queue(maxsize=100)  # {cam_id: gif_path, ...}
merge_buffer = {}
merge_lock = threading.Lock()
stop_event = threading.Event()


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeetNet(2).to(device)
model.load_state_dict(torch.load("../model/training/final_feet_net.pth", weights_only=True, map_location="cpu"))
model.eval()

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def start_ffmpeg_stream(rtsp_url):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-i",
            rtsp_url,
            "-loglevel",
            "quiet",
            "-vf",
            "fps=10",
            "-q:v",
            "5",
            "-f",
            "mjpeg",
            "-",
        ],
        stdout=subprocess.PIPE,
        bufsize=0,
    )


pil_transform = T.Compose(
    [
        T.Resize((112, 112)),
        T.ToTensor(),
    ]
)
norm_transform = T.Normalize((0.4316, 0.3945, 0.3765), (0.2280, 0.2215, 0.2170))


def run_model(cam1_tensor, cam2_tensor):
    # TODO: replace with your actual model
    
    # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: NONE
    out = model(cam1_tensor, cam2_tensor)
    probs = torch.nn.functional.softmax(out, dim=1)
    pred, index = torch.max(probs.data, 1)

    res = index.detach().int()

    print("Result: ", res)
    
    return res


# ─── THREADS ──────────────────────────────────────────────────────────────────
def camera_worker(cam_id, rtsp_url):
    while not stop_event.is_set():
        proc = start_ffmpeg_stream(rtsp_url)
        buffer = bytearray()
        frames = []
        print(f"[Camera{cam_id}] FFmpeg started.")
        try:
            while not stop_event.is_set():
                chunk = proc.stdout.read(4096)
                if not chunk:
                    raise IOError("EOF from FFmpeg")
                buffer.extend(chunk)
                while True:
                    soi = buffer.find(b"\xff\xd8")
                    eoi = buffer.find(b"\xff\xd9", soi + 2)
                    if soi < 0 or eoi < 0:
                        break
                    jpeg = bytes(buffer[soi : eoi + 2])
                    del buffer[: eoi + 2]
                    try:
                        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                    except:
                        continue
                    frames.append(img)
                    print(f"[Camera{cam_id}] frame {len(frames)}/{BUFFER_LEN}")
                    if len(frames) == BUFFER_LEN:
                        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                        try:
                            frame_queue.put_nowait((cam_id, list(frames), ts))
                            print(f"[Camera{cam_id}] enqueued @ {ts}")
                        except queue.Full:
                            print(f"[Camera{cam_id}] frame_queue FULL")
                        frames.clear()
        except Exception as e:
            print(f"[Camera{cam_id}] Error: {e} → restarting FFmpeg")
        finally:
            proc.kill()
    print(f"[Camera{cam_id}] exiting.")


def merging_worker():
    while not stop_event.is_set():
        try:
            cam_id, frames, ts = frame_queue.get(timeout=1.0)
            print(f"[Merger] pulled from frame_queue (size={frame_queue.qsize()})")
        except queue.Empty:
            continue

        with merge_lock:
            merge_buffer[cam_id] = (frames, ts)
            if set(merge_buffer) == set(RTSP_URLS):
                gifs = {}
                for cid, (frm, st) in merge_buffer.items():
                    path = os.path.join(OUTPUT_ROOT, f"cam_{cid}", f"cam{cid}_{st}.gif")
                    frm[0].save(
                        path, save_all=True, append_images=frm[1:], duration=100, loop=0
                    )
                    print(f"[Merger] wrote {path}")
                    gifs[cid] = path
                merge_buffer.clear()
                try:
                    inference_queue.put_nowait(gifs)
                    print(
                        f"[Merger] job→inference {gifs} (inference_queue size={inference_queue.qsize()})"
                    )
                except queue.Full:
                    print("[Merger] inference_queue FULL")
    print("[Merger] exiting.")


def processing_worker(wid):
    while not stop_event.is_set():
        # heartbeat when idle
        if inference_queue.qsize() == 0:
            print(f"[Infer{wid}] waiting for jobs (queue size=0)")
        try:
            gif_paths = inference_queue.get(timeout=1.0)
            print(f"[Infer{wid}] got job, queue size now {inference_queue.qsize()}")
        except Exception:
            continue

        try:
            cam_tensors = {}
            for cid, path in gif_paths.items():
                gif = Image.open(path)
                raw = [
                    frame.convert("RGB")
                    for i, frame in enumerate(ImageSequence.Iterator(gif))
                    if i < BUFFER_LEN
                ]
                gif.close()

                processed = []
                for im in raw:
                    arr = np.array(im)
                    crop_arr = arr[112 : 480 - 112, 224 : 704 - 224, :]
                    pil_crop = Image.fromarray(crop_arr)
                    t = pil_transform(pil_crop).to("cpu")
                    t = norm_transform(t)
                    processed.append(t)
                cam_tensors[cid] = torch.stack(processed, dim=0).unsqueeze(0)
                print(f"[Infer{wid}] cam{cid}: {cam_tensors[cid].shape}")

            code = run_model(cam_tensors[2], cam_tensors[3])
            print(f"[Infer{wid}] model→direction {code}")
            socket_server.send_direction(code)

        except Exception as e:
            print(f"[Infer{wid}] CRASHED on batch: {e}")
            traceback.print_exc()
            continue

    print(f"[Infer{wid}] exiting.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for cid in RTSP_URLS:
        os.makedirs(os.path.join(OUTPUT_ROOT, f"cam_{cid}"), exist_ok=True)

    socket_server.start()

    threading.Thread(target=merging_worker, daemon=True).start()
    for cid, url in RTSP_URLS.items():
        threading.Thread(target=camera_worker, args=(cid, url), daemon=True).start()
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
