#!/usr/bin/env python3
"""llav-set: set-level judge on Qwen3-Reranker-4B (MLX, LoRA). One training step = one question + a SET of candidates in a
single sequence; verdicts are read out at per-candidate slots placed AFTER all candidates, so every verdict attends to the
whole set (causal attention would otherwise hide later candidates from earlier ones).

Sequence  : <|im_start|>system … <|im_start|>user <Instruct>: … <Query>: q
            <Candidates>:
            [1] title: text(truncated) … [K] title: text
            <|im_end|><|im_start|>assistant <think></think> Verdicts: [1] [2] … [K]
Read-out  : hidden state at the "]" token closing each verdict slot "[k]" → lm_head → logit(yes) − logit(no).
Loss      : per-slot BCE(label) + λ · listwise CE (softmax over the K margins, targets = gold uniform) + optional count term.
Data      : rows from build-set.py ({q, items[{title,text,label}], src}); order shuffled every visit; sets capped at --max-k.
usage: train-set-judge.py --model DIR --sources a.jsonl[:N] ... --adapter-path OUT [--resume ADAPTERS] [--iters 3000] [--lr 2e-5]
       [--max-k 30] [--doc-tokens 110] [--max-seq-length 5120] [--lam 1.0] [--valid 150] [--save-every 250] [--steps-per-eval 250]"""
import argparse, json, math, os, random, time
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import mlx_lm.utils as _U; _U.MODEL_REMAPPING["qwen3_5_text"] = "qwen3_5"
from mlx_lm.utils import load, save_config
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.tuner.trainer import TrainingArgs, train

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--sources", nargs="+", required=True); ap.add_argument("--adapter-path", required=True)
ap.add_argument("--resume", default=""); ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--cosine", action="store_true"); ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--max-k", type=int, default=30); ap.add_argument("--min-k", type=int, default=6); ap.add_argument("--doc-tokens", type=int, default=110); ap.add_argument("--max-seq-length", type=int, default=5120)
ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--valid", type=int, default=150); ap.add_argument("--save-every", type=int, default=250); ap.add_argument("--steps-per-eval", type=int, default=250)
ap.add_argument("--num-layers", type=int, default=16); ap.add_argument("--rank", type=int, default=8); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--grad-checkpoint", action="store_true")
ap.add_argument("--instruct", default="Given a multi-hop question and a set of candidate passages, decide for EACH candidate whether it is necessary evidence in a fully supported answer chain (a bridge passage that identifies an intermediate entity counts). Candidates can only be judged together: a passage is necessary only if the chain it belongs to is consistent with the other selected passages.")
a = ap.parse_args(); random.seed(a.seed)
model, tok = load(a.model)
try: mx.set_cache_limit(6 << 30)
except Exception: pass
YES = tok.encode("yes", add_special_tokens=False)[0]; NO = tok.encode("no", add_special_tokens=False)[0]
SYSTEM = "Judge whether each candidate Document meets the requirements based on the Query and the Instruct provided. For every verdict slot the answer can only be \"yes\" or \"no\"."
PRE = tok.encode("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n" + f"<Instruct>: {a.instruct}\n<Query>: ", add_special_tokens=False)
MID = tok.encode("\n<Candidates>:\n", add_special_tokens=False)
SUF = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nVerdicts:", add_special_tokens=False)
SLOT = [tok.encode(f" [{k}]", add_special_tokens=False) for k in range(0, a.max_k + 1)]      # verdict slots; read at the last token of each
HEAD = [tok.encode(f"[{k}] ", add_special_tokens=False) for k in range(0, a.max_k + 1)]

def encode(q, items):
    """items: list of (title, text, label) already shuffled & capped. Returns (ids, slot_positions, labels)."""
    ids = PRE + tok.encode(q, add_special_tokens=False) + MID
    budget = a.max_seq_length - len(ids) - len(SUF) - sum(len(s) for s in SLOT[1:len(items) + 1]) - 8
    per = max(40, min(a.doc_tokens, budget // max(len(items), 1) - 6))
    for k, (title, text, _) in enumerate(items, 1):
        body = tok.encode(f"{title}: {text}", add_special_tokens=False)[:per]
        ids += HEAD[k] + body + tok.encode("\n", add_special_tokens=False)
    ids += SUF; pos = []
    for k in range(1, len(items) + 1):
        ids += SLOT[k]; pos.append(len(ids) - 1)
    return ids, pos, [int(l) for _, _, l in items]

# ---- data
rows = []
for spec in a.sources:
    path, _, lim = spec.partition(":"); rs = [json.loads(l) for l in open(os.path.expanduser(path))]; random.shuffle(rs)
    rows += rs[: int(lim)] if lim else rs
random.shuffle(rows)
print(f"rows {len(rows)} from {len(a.sources)} sources", flush=True)

def sample_set(r):
    items = [(i["title"], i["text"], i["label"]) for i in r["items"]]
    pos = [x for x in items if x[2]]; neg = [x for x in items if not x[2]]
    k = random.randint(max(a.min_k, len(pos) + 1), a.max_k) if len(items) > a.max_k else len(items)
    chosen = pos + random.sample(neg, min(len(neg), max(1, k - len(pos))))
    random.shuffle(chosen)
    return chosen[: a.max_k]

class SetData:
    def __init__(self, rs): self.rs = rs
    def __len__(self): return len(self.rs)
    def __getitem__(self, i): return encode(self.rs[i]["q"], sample_set(self.rs[i]))

valid_set = SetData(rows[: a.valid]); train_set = SetData(rows[a.valid:])

def iterate(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
    order = list(range(len(dataset)))
    while True:
        if loop: random.shuffle(order)
        for i in order:
            ids, pos, labels = dataset[i]
            K = len(pos); pad = a.max_k - K
            meta = mx.array([pos + [pos[-1]] * pad, labels + [0] * pad, [1] * K + [0] * pad])   # (3, max_k): slot positions, labels, mask
            yield mx.array([ids]), meta
        if not loop: break

def set_loss(model, batch, meta):
    tm = model.language_model if hasattr(model, "language_model") else model
    pos = meta[0]; labels = meta[1].astype(mx.float32); mask = meta[2].astype(mx.float32)   # fixed length max_k, padded slots masked
    hidden = tm.model(batch)[0]                                    # [T, H]
    h = hidden[pos]                                                # [max_k, H]  states at the verdict slots
    lg = tm.model.embed_tokens.as_linear(h) if getattr(tm.args, "tie_word_embeddings", False) else tm.lm_head(h)
    m = (lg[:, YES] - lg[:, NO]).astype(mx.float32)                # [max_k] margins
    n = mx.maximum(mx.sum(mask), 1.0)
    bce = mx.sum(nn.losses.binary_cross_entropy(m, labels, with_logits=True, reduction="none") * mask) / n
    neg_inf = mx.array(-1e9)
    lse_all = mx.logsumexp(mx.where(mask > 0.5, m, neg_inf)); lse_pos = mx.logsumexp(mx.where((labels > 0.5) & (mask > 0.5), m, neg_inf))
    listwise = lse_all - lse_pos
    return bce + a.lam * listwise, n

mx.random.seed(a.seed); model.freeze()
linear_to_lora_layers(model, a.num_layers, {"rank": a.rank, "dropout": 0.0, "scale": 20.0}, use_dora=False)
if a.resume: model.load_weights(a.resume, strict=False); print("resumed", a.resume, flush=True)
print_trainable_parameters(model); os.makedirs(a.adapter_path, exist_ok=True)
save_config({"fine_tune_type": "lora", "num_layers": a.num_layers, "lora_parameters": {"rank": a.rank, "dropout": 0.0, "scale": 20.0}, "model": a.model, "sources": a.sources, "max_k": a.max_k,
             "doc_tokens": a.doc_tokens, "iters": a.iters, "learning_rate": a.lr, "max_seq_length": a.max_seq_length, "objective": "set-level: per-slot BCE + listwise CE over the set"},
            os.path.join(a.adapter_path, "adapter_config.json"))
sched = optim.cosine_decay(a.lr, max(1, a.iters - a.warmup), a.lr * 0.1) if a.cosine else a.lr
if a.cosine and a.warmup: sched = optim.join_schedules([optim.linear_schedule(0.0, a.lr, a.warmup), sched], [a.warmup])
opt = optim.Adam(learning_rate=sched)
args = TrainingArgs(batch_size=1, iters=a.iters, val_batches=min(25, a.valid), steps_per_report=20, steps_per_eval=a.steps_per_eval, steps_per_save=a.save_every,
                    adapter_file=os.path.join(a.adapter_path, "adapters.safetensors"), max_seq_length=a.max_seq_length, grad_checkpoint=a.grad_checkpoint)
train(model=model, optimizer=opt, train_dataset=train_set, val_dataset=valid_set, args=args, loss=set_loss, iterate_batches=iterate)
print("done", flush=True)
