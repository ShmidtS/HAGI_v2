import os, sys
from huggingface_hub import HfApi

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
api = HfApi()
api.upload_large_folder(
    repo_id="Svyatoblood/HAGI-DeepSeek-V4-Flash-0731-2M",
    repo_type="model",
    folder_path="dsv4_release",
)
print("UPLOAD COMPLETE")
