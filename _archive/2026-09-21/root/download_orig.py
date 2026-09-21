"""Скачивание deepseek-ai/DeepSeek-V4-Flash-0731 (оригинал) в HF cache."""
import os, sys, time

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"

def main():
    from huggingface_hub import snapshot_download
    print(f"[{time.strftime('%H:%M:%S')}] start download {REPO}", flush=True)
    t0 = time.time()
    path = snapshot_download(
        repo_id=REPO,
        max_workers=8,
        tqdm_class=None,
    )
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] DONE in {dt/60:.1f} min -> {path}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        sys.exit(1)
