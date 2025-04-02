import os
import requests
from tqdm import tqdm

# Folder to save videos
os.makedirs("URFall/Videos", exist_ok=True)

# Base URL for video previews
BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data/"

# Fall numbers (01 to 13)
fall_ids = [f"{i:02d}" for i in range(1, 30)]
camera_ids = ["cam0", "cam1"]

# Create list of video URLs
video_urls = [
    f"{BASE_URL}fall-{fall_id}-{cam}.mp4"
    for fall_id in fall_ids
    for cam in camera_ids
]

def download_video(url):
    filename = url.split("/")[-1]
    filepath = os.path.join("URFall/Videos", filename)

    if os.path.exists(filepath):
        print(f"[✓] Already downloaded: {filename}")
        return

    print(f"[↓] Downloading: {filename}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(filepath, "wb") as f, tqdm(
            desc=filename,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

# Download all videos
for url in video_urls:
    download_video(url)
