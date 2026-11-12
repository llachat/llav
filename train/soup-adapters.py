#!/usr/bin/env python3
"""Average LoRA adapter checkpoints (uniform "soup") to reduce step-to-step variance.
usage: soup-adapters.py OUT_DIR ADAPTER_DIR STEP [STEP ...]   e.g. soup-adapters.py /tmp/ad-soup ~/models/rerank-ft/qwen3-rr-4b-lw4b-v1 0001250 0001500 0001750"""
import os, shutil, sys
import mlx.core as mx
out, ad, steps = sys.argv[1], os.path.expanduser(sys.argv[2]), sys.argv[3:]
os.makedirs(out, exist_ok=True); shutil.copy(os.path.join(ad, "adapter_config.json"), out)
acc = None
for s in steps:
    w = mx.load(os.path.join(ad, f"{s}_adapters.safetensors"))
    acc = {k: v.astype(mx.float32) for k, v in w.items()} if acc is None else {k: acc[k] + v.astype(mx.float32) for k, v in w.items()}
avg = {k: (v / len(steps)).astype(mx.bfloat16) if v.dtype != mx.float32 else v / len(steps) for k, v in acc.items()}
# keep original dtype
ref = mx.load(os.path.join(ad, f"{steps[0]}_adapters.safetensors"))
avg = {k: (acc[k] / len(steps)).astype(ref[k].dtype) for k in acc}
mx.save_safetensors(os.path.join(out, "adapters.safetensors"), avg)
print(f"soup of {steps} -> {out} ({len(avg)} tensors)")
