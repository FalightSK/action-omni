"""One-time download: lerobot/pusht into the project (asset/data/pusht),
not the HF hub cache — keeps all real data inside the repo tree."""
from pathlib import Path
from huggingface_hub import snapshot_download

local_dir = str(Path(__file__).parents[2] / "asset" / "data" / "pusht")
print("Downloading lerobot/pusht ...")
path = snapshot_download(
    repo_id="lerobot/pusht",
    repo_type="dataset",
    revision="v3.0",
    local_dir=local_dir,
)
print("Downloaded to:", path)
