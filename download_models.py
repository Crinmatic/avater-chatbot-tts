import os
import requests
from pathlib import Path
from huggingface_hub import snapshot_download

HF_TOKEN = os.getenv("HF_TOKEN")

MODELS = {
    "fa_IR-amir-medium": {
        "base_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium",
        "files": ["fa_IR-amir-medium.onnx", "fa_IR-amir-medium.onnx.json"]
    },
}

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    try:
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 404:
             print(f"404 Not Found: {url}")
             return False
        response.raise_for_status()
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    base_models_dir = Path("models")

    yoruba_cache_dir = base_models_dir / "hf-cache"
    print(f"Downloading official Yoruba MMS model into {yoruba_cache_dir}...")
    snapshot_download(
        repo_id="facebook/mms-tts-yor",
        cache_dir=str(yoruba_cache_dir),
        local_dir=str(yoruba_cache_dir / "mms-tts-yor"),
        local_dir_use_symlinks=False,
    )

    for name, info in MODELS.items():
        base_url = info["base_url"]
        
        # Normalize target directory
        if name.startswith("mms-yor"):
             target_dir = base_models_dir / "vits-mms-yor"
        else:
             target_dir = base_models_dir / f"vits-{name.replace('_', '-')}"
        
        for filename in info["files"]:
            url = f"{base_url}/{filename}"
            # Rename vocab.txt to tokens.txt if needed
            local_filename = "tokens.txt" if filename == "vocab.txt" else filename
            target_path = target_dir / local_filename
            
            if not target_path.exists():
                success = download_file(url, target_path)
                if not success and name == "mms-yor-tokens":
                    # Try alternative for yor vocab
                    alt_url = f"https://huggingface.co/facebook/mms-tts-yor/raw/main/vocab.txt"
                    print(f"Retrying with {alt_url}...")
                    download_file(alt_url, target_path)
            else:
                print(f"File already exists: {target_path}")

if __name__ == "__main__":
    main()
