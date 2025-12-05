from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/sam3",
    repo_type="model",                       # optional; defaults to "model"
    local_dir="models/sam3",      # directory where files are stored
    local_dir_use_symlinks=False             # ensure no symlinks, replicate files
)
