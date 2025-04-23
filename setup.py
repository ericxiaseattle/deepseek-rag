import os
from huggingface_hub import snapshot_download


# Download the 1.73b, 2.22b, and 2.51b quantizations of DeepSeek-R1
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
snapshot_download(
  repo_id = "unsloth/DeepSeek-R1-GGUF",
  local_dir = "DeepSeek-R1-GGUF",
  allow_patterns = [
    # "*UD-IQ1_S*",
    "*UD-IQ1_M*",
    "*UD-IQ2_XXS*",
    "*UD-Q2_K_XL*"
]
)
