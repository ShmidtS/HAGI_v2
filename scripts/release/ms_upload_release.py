"""Upload dsv4_release to ModelScope (www.modelscope.ai), resumable."""
import os, sys, time

ENDPOINT = "https://www.modelscope.ai"
REPO = "svyatoblood/HAGI-DeepSeek-V4-Flash-0731-2M"
FOLDER = "dsv4_release"

def load_token():
    for line in open(".env"):
        line = line.strip()
        if line.startswith("MODELSCOPE_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("MODELSCOPE_TOKEN not found in .env")

def main():
    from modelscope.hub.api import HubApi
    api = HubApi(endpoint=ENDPOINT)
    api.login(load_token())
    try:
        api.create_model(REPO)
        print("repo created:", REPO, flush=True)
    except Exception as e:
        print("create_model:", type(e).__name__, str(e)[:200], flush=True)
    t0 = time.time()
    res = api.upload_folder(
        repo_id=REPO,
        repo_type="model",
        folder_path=FOLDER,
        commit_message="HAGI terni4 release: 43 layer banks + routers + skeleton + tokenizer (102 GB)",
        ignore_patterns=["__pycache__/**", ".cache/**", "**/__pycache__/**"],
        max_workers=4,
        use_cache=True,
    )
    print("upload done in %.1f min, result: %s" % ((time.time() - t0) / 60, str(res)[:500]), flush=True)

if __name__ == "__main__":
    sys.exit(main())
