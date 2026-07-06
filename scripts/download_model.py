"""Download google/gemma-4-E2B-it using HF_TOKEN from .env."""

import os

env_path = os.path.join(os.path.dirname(__file__), ".env")
token = None
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            token = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

if not token:
    print("ERROR: HF_TOKEN not found in .env")
    exit(1)

print(f"Token: {token[:10]}...")
os.environ["HF_TOKEN"] = token

from huggingface_hub import snapshot_download  # noqa: E402

print("Downloading google/gemma-4-E2B-it...")
path = snapshot_download("google/gemma-4-E2B-it", token=token)
path2 = snapshot_download("google/t5gemma-2-1b-1b", token=token)
print(f"Downloaded to: {path}, {path2}")
