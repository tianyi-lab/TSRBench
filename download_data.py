from huggingface_hub import snapshot_download

snapshot_download(repo_id="ParadiseYu/TSRBench", local_dir="./dataset", local_dir_use_symlinks=False, repo_type="dataset")
