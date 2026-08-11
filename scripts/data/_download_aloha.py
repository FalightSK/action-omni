"""One-time download: lerobot/aloha_sim_transfer_cube_human into the project (asset/data/),
not the HF hub cache — keeps all real data inside the repo tree."""
from pathlib import Path
from huggingface_hub import snapshot_download

local_dir = str(Path(__file__).parents[2] / "asset" / "data" / "aloha_sim_transfer_cube_human")
print("Downloading lerobot/aloha_sim_transfer_cube_human ...")
path = snapshot_download(
    repo_id="lerobot/aloha_sim_transfer_cube_human",
    repo_type="dataset",
    local_dir=local_dir,
)
print("Downloaded to:", path)
