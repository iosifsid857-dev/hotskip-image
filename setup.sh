#!/usr/bin/env bash
# Downloads the official Tencent code + NF4 weights (~51 GB) and creates a venv.
set -euo pipefail
cd "$(dirname "$0")"

OFFICIAL=tencent/HunyuanImage-3.0-Instruct-Distil
WEIGHTS=${HY3_WEIGHTS_REPO:-EricRollei/HunyuanImage-3.0-Instruct-Distil-NF4-v2}

python3 -m venv .venv
. .venv/bin/activate
pip install -q torch torchvision transformers diffusers bitsandbytes safetensors einops pillow numpy "huggingface_hub[cli]"

# Official model code from Tencent (we only import it, nothing is modified)
mkdir -p hy3pkg
for f in modeling_hunyuan_image_3.py configuration_hunyuan_image_3.py hunyuan_image_3_pipeline.py \
         autoencoder_kl_3d.py image_processor.py tokenization_hunyuan_image_3.py siglip2.py \
         cache_utils.py system_prompt.py; do
  curl -sfL -o "hy3pkg/$f" "https://huggingface.co/$OFFICIAL/resolve/main/$f"
done
touch hy3pkg/__init__.py

# NF4 weights (~51 GB). Only *.safetensors + json/tokenizer files are used, no code from this repo is executed.
hf download "$WEIGHTS" --local-dir weights --include "*.safetensors" "*.json"
echo "done. Try:  .venv/bin/python hy3run.py 'a red fox sitting in snow, photo' fox.png"
