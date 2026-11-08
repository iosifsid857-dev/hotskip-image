# HOTskip-Image: HunyuanImage-3.0 (80B MoE) on an 8 GB GPU. Layers and experts are streamed per layer.
# Usage: python hy3run.py "<prompt>" [out.png] [steps] [WxH]   (env HY3_TAYLOR=2,1,2,1 enables TaylorCache)
import sys, os, time, json, threading, queue
import torch, torch.nn.functional as F
from safetensors import safe_open
import bitsandbytes.functional as BF

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.environ.get("HY3_WEIGHTS", os.path.join(HERE, "weights"))
sys.path.insert(0, HERE)   # hy3pkg -> official Tencent code (see setup.sh)
import hy3pkg.configuration_hunyuan_image_3 as C
import hy3pkg.modeling_hunyuan_image_3 as Mm

DEV = torch.device("cuda")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

IDX = json.load(open(W + "/model.safetensors.index.json"))["weight_map"]
_files = {f: safe_open(W + "/" + f, "pt") for f in set(IDX.values())}
def get(name): return _files[IDX[name]].get_tensor(name)

cfg = C.HunyuanImage3Config.from_pretrained(W)
cfg._attn_implementation = "sdpa"
cfg.moe_impl = "eager"
if not hasattr(cfg, "model_version"): cfg.model_version = "HunyuanImage-3.0-Instruct-Distil"
with torch.device("meta"):
    model = Mm.HunyuanImage3ForCausalMM(cfg)
Mm.HunyuanImage3ForCausalMM.device = property(lambda self: DEV)
import transformers.cache_utils as _cu
_li = _cu.StaticLayer.lazy_initialization
def _lazy(self, k, v=None):
    _li(self, k, k if v is None else v)
_cu.StaticLayer.lazy_initialization = _lazy

def set_param(mod_name, pname, t):
    mod = model.get_submodule(mod_name) if mod_name else model
    mod._parameters[pname] = torch.nn.Parameter(t, requires_grad=False)

layers = model.model.layers
NL = len(layers)
layer_cpu = [dict() for _ in range(NL)]   # (mod, pname) -> pinned cpu tensor

for name, p in list(model.named_parameters()):
    if ".experts." in name:
        continue
    mod_name, pname = name.rsplit(".", 1)
    t = get(name)
    if name.startswith("model.layers."):
        li = int(name.split(".")[2])
        t = t.to(torch.bfloat16 if t.is_floating_point() and "gate.wg" not in name else t.dtype).pin_memory()
        layer_cpu[li][(mod_name, pname)] = t
        set_param(mod_name, pname, t)
    elif name.startswith(("vae.", "vision_model.", "vision_aligner.", "model.wte.", "lm_head.")):
        set_param(mod_name, pname, t)
    else:
        set_param(mod_name, pname, t.to(DEV))
log("resident weights loaded")

def cpu_module_on_gpu(mod):
    f = mod.forward
    def fwd(*a, **k):
        a = [x.cpu() if torch.is_tensor(x) else x for x in a]
        return f(*a, **k).to(DEV)
    mod.forward = fwd
cpu_module_on_gpu(model.model.wte)
cpu_module_on_gpu(model.lm_head)
_vae_decode = model.vae.decode
def vae_decode(z, *a, **k):
    torch.save(z.detach().cpu(), "last_latents.pt")
    log("latents saved, VAE decode...", tuple(z.shape))
    global gpu_slot
    gpu_slot = None
    torch.cuda.empty_cache()
    model.vae.to(DEV, torch.float16)
    model.vae.enable_spatial_tiling(True)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            return _vae_decode(z.to(DEV, torch.float16), *a, **k)
    finally:
        model.vae.to("cpu", torch.float32); torch.cuda.empty_cache()
model.vae.decode = vae_decode

copy_stream = torch.cuda.Stream()
on_gpu = {}   # li -> (event, {key: gpu tensor})
def start_layer_copy(li):
    if li >= NL or li in on_gpu: return
    with torch.cuda.stream(copy_stream):
        d = {k: v.to(DEV, non_blocking=True) for k, v in layer_cpu[li].items()}
        ev = torch.cuda.Event(); ev.record()
    on_gpu[li] = (ev, d)
def pre_hook(li):
    def h(mod, args, kwargs=None):
        start_layer_copy(li)
        ev, d = on_gpu[li]
        torch.cuda.current_stream().wait_event(ev)
        for (mn, pn), t in d.items():
            model.get_submodule(mn)._parameters[pn].data = t
        start_layer_copy(li + 1)
        expert_prefetch(li + 1)
    return h
def post_hook(li):
    def h(mod, args, out):
        for (mn, pn), t in layer_cpu[li].items():
            model.get_submodule(mn)._parameters[pn].data = t
        on_gpu.pop(li, None)
    return h
for li, L in enumerate(layers):
    L.register_forward_pre_hook(pre_hook(li))
    L.register_forward_hook(post_hook(li))

NE = cfg.num_experts
PARTS = ("gate_and_up_proj", "down_proj")
meta = {}
def expert_meta(li, e, part):
    key = (li, e, part)
    if key not in meta:
        b = f"model.layers.{li}.mlp.experts.{e}.{part}.weight"
        js = json.loads(bytes(get(b + ".quant_state.bitsandbytes__nf4").numpy()).decode())
        meta[key] = (torch.Size(js["shape"]), js["nested_offset"],
                     get(b + ".nested_absmax").to(DEV), get(b + ".nested_quant_map").to(DEV),
                     get(b + ".quant_map").to(DEV))
    return meta[key]

import struct
from concurrent.futures import ThreadPoolExecutor
_hdr = {}
for f in set(IDX.values()):
    with open(W + "/" + f, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]; _hdr[f] = (8 + n, json.loads(h.read(n)))
_fd = {f: os.open(W + "/" + f, os.O_RDONLY) for f in _hdr}
def _loc(name):
    f = IDX[name]; base, hd = _hdr[f]; o = hd[name]["data_offsets"]
    return _fd[f], base + o[0], o[1] - o[0]
LAYOUT, SLOT = [], 0
for li in range(NL):
    items, off = [], 0
    for e in range(NE):
        for part in PARTS:
            for kind, suf in (("w", ""), ("a", ".absmax")):
                fd, fo, ln = _loc(f"model.layers.{li}.mlp.experts.{e}.{part}.weight{suf}")
                items.append((e, part, kind, fd, fo, off, ln)); off += (ln + 511) // 512 * 512
    LAYOUT.append(items); SLOT = max(SLOT, off)
NSLOT = 3
cpu_slots = [torch.empty(SLOT, dtype=torch.uint8).pin_memory() for _ in range(NSLOT)]
gpu_slot = None
log(f"expert layer buffer {SLOT/2**30:.2f} GiB x {NSLOT} (pinned RAM) + 1 (GPU)")
_pool = ThreadPoolExecutor(int(os.environ.get("HY3_IO", "8")))
_free_slots = queue.Queue()
for i in range(NSLOT): _free_slots.put(i)
_pending = {}   # li -> (slot, [futures])
def _pread(fd, fo, buf, ln):
    mv = memoryview(buf)
    got = 0
    while got < ln:
        got += os.preadv(fd, [mv[got:ln]], fo + got)
def expert_prefetch(li):
    li %= NL
    if li in _pending: return
    try: si = _free_slots.get_nowait()
    except queue.Empty: return
    arr = cpu_slots[si].numpy()
    futs = [_pool.submit(_pread, fd, fo, arr[so:so + ln], ln) for (_, _, _, fd, fo, so, ln) in LAYOUT[li]]
    _pending[li] = (si, futs)
def expert_take(li):
    expert_prefetch(li)
    if li not in _pending:
        raise RuntimeError("no free slot")
    si, futs = _pending.pop(li)
    for f in futs: f.result()
    global gpu_slot
    if gpu_slot is None: gpu_slot = torch.empty(SLOT, dtype=torch.uint8, device=DEV)
    gpu_slot.copy_(cpu_slots[si], non_blocking=False)
    _free_slots.put(si)
    expert_prefetch(li + 1); expert_prefetch(li + 2)
    out = {}
    for (e, part, kind, fd, fo, so, ln) in LAYOUT[li]:
        out.setdefault((e, part), {})[kind] = gpu_slot[so:so + ln]
    return {k: (v["w"], v["a"]) for k, v in out.items()}

def dequant(li, e, part, packed, absmax):
    shape, off, nabs, nqm, qm = expert_meta(li, e, part)
    st2 = BF.QuantState(absmax=nabs, code=nqm, blocksize=256, dtype=torch.float32)
    qs = BF.QuantState(absmax=absmax, shape=shape, code=qm, blocksize=64,
                       quant_type="nf4", dtype=torch.bfloat16, offset=torch.tensor(off, device=DEV), state2=st2)
    return BF.dequantize_4bit(packed.view(-1, 1), qs)

STATS = {"moe_s": 0.0, "wait_s": 0.0}
def moe_forward(self, hidden_states):
    bsz, seq, H = hidden_states.shape
    li = self.layer_idx
    shared = self.shared_mlp(hidden_states)
    x = hidden_states.reshape(-1, H)
    with torch.autocast("cuda", enabled=False):
        w, idx = self.gate(hidden_states, topk_impl="easy")
    w = w.to(x.dtype)
    t = time.time(); bufs = expert_take(li); STATS["wait_s"] += time.time() - t
    t = time.time()
    out = torch.zeros_like(x)
    flat = idx.reshape(-1)
    tok = torch.arange(x.shape[0], device=x.device).repeat_interleave(idx.shape[1])
    wf = w.reshape(-1)
    order = torch.argsort(flat)
    counts = torch.bincount(flat, minlength=NE).tolist()
    start = 0
    for e in range(NE):
        n = counts[e]
        if n == 0: continue
        sel = order[start:start + n]; start += n
        ti = tok[sel]
        gu = dequant(li, e, "gate_and_up_proj", *bufs[(e, "gate_and_up_proj")])
        h = F.linear(x[ti], gu); del gu
        x1, x2 = h.chunk(2, dim=-1)
        dn = dequant(li, e, "down_proj", *bufs[(e, "down_proj")])
        y = F.linear(x1 * F.silu(x2), dn); del dn
        out.index_add_(0, ti, y * wf[sel, None])
    torch.cuda.synchronize()
    STATS["moe_s"] += time.time() - t
    return shared + out.reshape(bsz, seq, H)
Mm.HunyuanMoE.forward = moe_forward

model.eval()
for k, v in json.load(open(W + "/generation_config.json")).items(): setattr(model.generation_config, k, v)
model.load_tokenizer(W)
log("ready, VRAM", round(torch.cuda.memory_allocated() / 2**30, 2), "GiB")

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a red fox sitting in snow, photo"
    outp = sys.argv[2] if len(sys.argv) > 2 else "out.png"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    size = sys.argv[4] if len(sys.argv) > 4 else "1024x1024"
    taylor = {}
    if os.environ.get("HY3_TAYLOR"):
        iv, od, fe, te = (int(x) for x in os.environ["HY3_TAYLOR"].split(","))
        taylor = dict(use_taylor_cache=True, taylor_cache_interval=iv, taylor_cache_order=od,
                      taylor_cache_enable_first_enhance=fe > 0, taylor_cache_first_enhance_steps=fe,
                      taylor_cache_enable_tailing_enhance=te > 0, taylor_cache_tailing_enhance_steps=te,
                      taylor_cache_low_freqs_order=0, taylor_cache_high_freqs_order=od)
        log("TaylorCache", taylor)
    t = time.time()
    with torch.no_grad():
        cot, samples = model.generate_image(prompt=prompt, seed=42, image_size=size, use_system_prompt="dynamic",
                                            bot_task="image", diff_infer_steps=steps, verbose=1, **taylor)
    samples[0].save(outp)
    log(f"image {outp} in {time.time()-t:.0f} s; MoE {STATS['moe_s']:.0f} s, waiting for disk {STATS['wait_s']:.0f} s",
        "peak VRAM", round(torch.cuda.max_memory_allocated() / 2**30, 2), "GiB")
