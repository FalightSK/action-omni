"""One-time download: lerobot/aloha_sim_transfer_cube_human to local lerobot cache."""
from huggingface_hub import snapshot_download

local_dir = r"C:\Users\SK\.cache\huggingface\lerobot\lerobot\aloha_sim_transfer_cube_human"
print("Downloading lerobot/aloha_sim_transfer_cube_human ...")
path = snapshot_download(
    repo_id="lerobot/aloha_sim_transfer_cube_human",
    repo_type="dataset",
    local_dir=local_dir,
)
print("Downloaded to:", path)
