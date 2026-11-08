# HOTskip-Image

**HunyuanImage-3.0 — the largest open image model (80B MoE) — on an 8 GB laptop GPU.**

Tencent recommends **≥ 3 × 80 GB** of VRAM (≥ 8 × 80 GB for the Instruct version).
HOTskip-Image runs the Instruct-Distil model on an **RTX 4070 Laptop (8 GB) + 32 GB RAM**:

| | |
|---|---|
| Model | HunyuanImage-3.0-Instruct-Distil, 80B total / 13B active, 64 experts × 32 layers, NF4 (48 GB) |
| Hardware | RTX 4070 Laptop 8 GB, 32 GB DDR5, NVMe SSD |
| Peak VRAM | **4.7 GB** |
| 1024×1024, 8 steps | **~78 s** (first naive version: 200 s) |

![fox](assets/fox.png)

*"a red fox sitting in snow, photo" — generated on the laptop above.*

[Русская версия](README.ru.md) · sister project: [HOTskip](https://github.com/iosifsid857-dev/hotskip) (MoE LLMs bigger than RAM+VRAM)

## How it works

A diffusion step pushes ~4400 tokens through the model, so almost **every** expert is used on every step —
all 40 GB of expert weights have to pass through the GPU 8 times per image. The trick is to make that stream as cheap as possible:

1. **Resident on GPU:** only small parts (patch/time embeddings, final layer) ≈ 1 GB.
2. **Per-layer streaming:** attention + shared MLP of a layer (bf16, pinned RAM) are copied on a side CUDA stream while the previous layer computes.
3. **Expert stream:** expert NF4 weights are read straight from the safetensors files with many parallel `pread`s into pinned ring buffers (3 layers ahead), then one big copy per layer to the GPU; dequantised one expert at a time. The OS page cache keeps whatever fits in RAM.
4. **VAE** runs on the GPU only at the end (fp16, tiled) after the expert buffer is freed.
5. **TaylorCache** (built into the model): some steps are predicted instead of computed — a skipped step reads no expert weights at all.

| version | s / image | disk wait | MoE compute |
|---|---|---|---|
| v1: one reader thread, double copies | 200 | 109 s | 76 s |
| parallel pread → pinned ring → one GPU copy | 95 | — | — |
| + TaylorCache (2,1,2,1) | **78** | 27 s | 34 s |

Output with the fast reader is identical to v1 (mean pixel difference 0.5/255).

## Run

```bash
./setup.sh        # venv, official Tencent code, NF4 weights (~51 GB download)
HY3_TAYLOR=2,1,2,1 .venv/bin/python hy3run.py "a red fox sitting in snow, photo" fox.png 8 1024x1024
```

Env: `HY3_WEIGHTS` (weights dir), `HY3_IO` (reader threads, default 8), `HY3_TAYLOR=interval,order,first_full,last_full`.
Needs ~6 GB free VRAM, ~10 GB free RAM (more RAM = more page cache = faster).

Tested with torch 2.14, transformers 5.17, diffusers 0.40, bitsandbytes 0.50 on Fedora 44.
Small runtime patches make the official code work with transformers 5 (StaticCache init, generation config, model_version).

## Notes

- Text-to-image only (`bot_task="image"`); the "think/recaption" text stage is not used yet.
- The model is Tencent's ([license](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil)); this repo contains only the loader (MIT). NF4 weights by [EricRollei](https://huggingface.co/EricRollei/HunyuanImage-3.0-Instruct-Distil-NF4-v2).

## Next

- keep the hottest experts resident in pinned RAM / VRAM instead of relying on page cache
- several images per weight pass (batch) — almost free extra images
- skip experts that receive very few tokens (HOTskip idea)
