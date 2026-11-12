#!/usr/bin/env python3
"""Group-wise (listwise) LoRA fine-tune of a Qwen3-Reranker for SAG2 evidence ranking, with mlx-lm's trainer but our own
batches and loss. Each training step = ONE question's candidate group (positives + sampled negatives); the loss is
  pointwise CE on (no, yes) logits at the label position  +  λ · listwise CE: −log( Σ_pos e^{m_i} / Σ_all e^{m_j} ),  m = logit_yes − logit_no
so the model learns both "is this evidence?" (calibration) and "evidence must outrank the other candidates" (the metric).

usage: train-rerank-listwise.py --model DIR --sources a.jsonl[:N] ... --adapter-path OUT [--group 8] [--lam 1.0] [--epochs 1]
       [--lr 1e-4] [--num-layers 16] [--rank 8] [--max-seq-length 1024] [--resume adapters.safetensors] [--valid 300]
Sources are Laya-format rows ({state, questions, expected}); grouping key = the state minus the candidate part."""
import argparse, json, os, random, re, time
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import mlx_lm.utils as _U; _U.MODEL_REMAPPING["qwen3_5_text"] = "qwen3_5"  # Qwen3.5 text-only checkpoints (Prism, Kev bases)
from mlx_lm.utils import load, save_config
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.tuner.trainer import TrainingArgs, train

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--sources", nargs="+", required=True); ap.add_argument("--adapter-path", required=True)
ap.add_argument("--soft", type=float, default=0.0, help="weight of a BCE term towards the teacher (Jev) score carried in row['jev'] (0 = off)"); ap.add_argument("--group", type=int, default=8); ap.add_argument("--iters", type=int, default=0); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--num-layers", type=int, default=16); ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--max-seq-length", type=int, default=1024); ap.add_argument("--steps-per-eval", type=int, default=400); ap.add_argument("--save-every", type=int, default=400)
ap.add_argument("--valid", type=int, default=300); ap.add_argument("--grad-checkpoint", action="store_true"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default=""); ap.add_argument("--cosine", action="store_true", help="warmup + cosine decay to 10%% of lr"); ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--max-chars", type=int, default=2400)
ap.add_argument("--instruct", default="Given a multi-hop question, judge whether the passage states a fact that is necessary to answer it (including a fact that identifies an intermediate entity in the answer chain).")
a = ap.parse_args(); random.seed(a.seed)
PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RX = re.compile(r"Original question: (.*?)\n(?:Sub-questions:\n((?:\d+\) .*\n)+))?(?:Previously retrieved evidence \(context only; may be wrong\):\n((?:\[\d+\] .*\n)+))?Candidate title: (.*?)\nCandidate passage: (.*)\Z", re.S)

model, tokenizer = load(a.model)
try: mx.set_cache_limit(4 << 30)
except Exception: pass
YES = tokenizer.encode("yes", add_special_tokens=False)[0]; NO = tokenizer.encode("no", add_special_tokens=False)[0]

def parse(state):
    m = RX.match(state)
    if not m: return None
    q, subq, ev, title, passage = m.groups(); query = q.strip()
    if subq: query += "\nSub-questions: " + " | ".join(l.strip() for l in subq.strip().split("\n"))
    if ev: query += "\nAlready retrieved evidence (context only): " + " || ".join(l.strip()[:300] for l in ev.strip().split("\n"))
    return q.strip(), query, f"{title.strip()}\n{passage.strip()}"

# ---- build groups
groups = {}
for spec in a.sources:
    path, _, n = spec.partition(":"); n = int(n) if n else 0
    rows = [json.loads(l) for l in open(os.path.expanduser(path))]; random.shuffle(rows)
    if n: rows = rows[:n]
    for r in rows:
        p = parse(r["state"])
        if not p: continue
        qkey, query, doc = p
        groups.setdefault(qkey, []).append((query, doc[:a.max_chars], bool(r["expected"]["verdict"]), int(round(float(r["jev"]) * 1000)) if isinstance(r.get("jev"), (int, float)) else -1))  # grouped by question; each item keeps its own context
keys = [k for k, v in groups.items() if any(x[2] for x in v) and any(not x[2] for x in v)]
random.shuffle(keys)
print(f"groups with both labels: {len(keys)} (of {len(groups)})", flush=True)

def encode_group(_qkey, items):
    pos = [(qq, d, sf) for qq, d, l, sf in items if l]; neg = [(qq, d, sf) for qq, d, l, sf in items if not l]
    k_pos = min(len(pos), max(1, a.group // 2)); k_neg = min(len(neg), a.group - k_pos)
    chosen = [(qq, d, 1, sf) for qq, d, sf in random.sample(pos, k_pos)] + [(qq, d, 0, sf) for qq, d, sf in random.sample(neg, k_neg)]
    random.shuffle(chosen); out = []
    for query, d, l, sf in chosen:
        ids = tokenizer.encode(PREFIX + f"<Instruct>: {a.instruct}\n<Query>: {query}\n<Document>: {d}" + SUFFIX, add_special_tokens=False)
        if len(ids) >= a.max_seq_length: ids = ids[: a.max_seq_length - 40] + ids[-39:]
        out.append((ids + [YES if l else NO], len(ids), sf))
    return out

class GroupSet:
    def __init__(self, ks): self.ks = ks
    def __len__(self): return len(self.ks)
    def __getitem__(self, i): return encode_group(self.ks[i], groups[self.ks[i]])  # fresh negative sample each visit

valid_set = GroupSet(keys[: a.valid]); train_set = GroupSet(keys[a.valid:])

def iterate_groups(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
    """One question group per step (mlx-lm trainer signature); batch_size is ignored, group size is --group."""
    order = list(range(len(dataset)))
    while True:
        if loop: random.shuffle(order)
        for i in order:
            items = dataset[i]
            L = max(len(t) for t, _, _ in items)
            arr = mx.array([t + [0] * (L - len(t)) for t, _, _ in items]); lens = mx.array([[off, len(t), sf] for t, off, sf in items])
            yield arr, lens
        if not loop: break

def judge_loss(model, batch, lengths):
    K = batch.shape[0]
    tm = model.language_model if hasattr(model, "language_model") else model
    hidden = tm.model(batch[:, :-1])                    # [K, T-1, H]
    pos = lengths[:, 0] - 1                             # position that predicts the label token
    h = hidden[mx.arange(K), pos]                       # [K, H]
    sel = tm.model.embed_tokens.as_linear(h) if getattr(tm.args, "tie_word_embeddings", False) else tm.lm_head(h)   # [K, V] only at label positions
    s_no, s_yes = sel[:, NO], sel[:, YES]
    labels = (batch[mx.arange(K), lengths[:, 0]] == YES).astype(mx.int32)
    pointwise = nn.losses.cross_entropy(mx.stack([s_no, s_yes], axis=1), labels, reduction="mean")
    m = (s_yes - s_no).astype(mx.float32)
    lse_all = mx.logsumexp(m)
    lse_pos = mx.logsumexp(mx.where(labels == 1, m, mx.array(-1e9)))
    listwise = lse_all - lse_pos
    loss = pointwise + a.lam * listwise
    if a.soft > 0:   # teacher (Jev) soft target: BCE(sigmoid(yes-no margin), jev_score) on rows that carry one
        sf = lengths[:, 2]; has = (sf >= 0).astype(mx.float32); t = mx.clip(sf.astype(mx.float32) / 1000.0, 0.0, 1.0)
        bce = nn.losses.binary_cross_entropy(m, t, with_logits=True, reduction="none")
        loss = loss + a.soft * mx.sum(bce * has) / mx.maximum(mx.sum(has), 1.0)
    return loss, mx.array(K)

iters = a.iters or int(a.epochs * len(train_set))
mx.random.seed(a.seed); model.freeze()
lora_params = {"rank": a.rank, "dropout": 0.0, "scale": 20.0}
linear_to_lora_layers(model, a.num_layers, lora_params, use_dora=False)
if a.resume: model.load_weights(a.resume, strict=False); print("resumed", a.resume)
print_trainable_parameters(model)
os.makedirs(a.adapter_path, exist_ok=True)
save_config({"fine_tune_type": "lora", "num_layers": a.num_layers, "lora_parameters": lora_params, "model": a.model, "sources": a.sources, "group": a.group, "lam": a.lam,
             "iters": iters, "learning_rate": a.lr, "max_seq_length": a.max_seq_length, "objective": "pointwise CE + listwise CE over question groups"},
            os.path.join(a.adapter_path, "adapter_config.json"))
args = TrainingArgs(batch_size=a.group, iters=iters, val_batches=min(len(valid_set), 60), steps_per_report=20, steps_per_eval=a.steps_per_eval,
                    steps_per_save=a.save_every, adapter_file=os.path.join(a.adapter_path, "adapters.safetensors"), max_seq_length=a.max_seq_length, grad_checkpoint=a.grad_checkpoint)
print(f"training {iters} group-steps (group {a.group}, λ={a.lam}), lr {a.lr}", flush=True)
lr_sched = a.lr
if a.cosine:
    from mlx_lm.tuner.utils import build_schedule
    lr_sched = build_schedule({"name": "cosine_decay", "arguments": [a.lr, max(iters - a.warmup, 1), a.lr * 0.1], "warmup": a.warmup, "warmup_init": 0.0})
    print(f"cosine schedule: warmup {a.warmup}, decay over {iters - a.warmup} steps to {a.lr*0.1:g}", flush=True)
train(model=model, optimizer=optim.Adam(learning_rate=lr_sched), train_dataset=train_set, val_dataset=valid_set, args=args, loss=judge_loss, iterate_batches=iterate_groups)
print("done", flush=True)
