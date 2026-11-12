#!/usr/bin/env python3
"""MLX-native judge benchmark for a (fused) Qwen3-Reranker: same 50 questions / 100 BM25 candidates, P(yes) from the
yes/no logits at each prompt's last position. Right padding + gather at the real last token, so causal attention is unaffected.
usage: mlx-rerank-bench.py --model-path DIR --details DETAILS.jsonl [--batch 16] [--max-len 1024] [--tag T] [--decomp oracle|file.json] [--limit N]"""
import argparse, json, os, statistics as st, time
import mlx.core as mx
import mlx_lm.utils as _U; _U.MODEL_REMAPPING["qwen3_5_text"] = "qwen3_5"  # Qwen3.5 text-only checkpoints (Prism, Kev bases)
from mlx_lm.utils import load
ap = argparse.ArgumentParser()
ap.add_argument("--model-path", required=True); ap.add_argument("--details", required=True); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--max-len", type=int, default=1024); ap.add_argument("--tag", default=""); ap.add_argument("--id-shift", type=int, default=0, help="SAG2 trace ids are 1-based chunk rowids; -1 maps them to exact corpus indices (default 0 keeps the historical candidate sets)"); ap.add_argument("--dump", default="", help="write per-question candidate scores as JSONL"); ap.add_argument("--decomp", default="")
ap.add_argument("--instruct", default="Given a multi-hop question, judge whether the passage states a fact that is necessary to answer it (including a fact that identifies an intermediate entity in the answer chain).")
ap.add_argument("--dataset-dir", default=os.path.expanduser("~/models/evals/sag2/bench/SAG-Benchmark/sag2/dataset"))
ap.add_argument("--adapter-path", default="", help="LoRA adapter dir to apply on top of the base")
ap.add_argument("--system", default="Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".")
a = ap.parse_args()
corpus = json.load(open(os.path.join(a.dataset_dir, "musique_corpus.json")))
key2id = {(c["title"], c["text"]): i for i, c in enumerate(corpus)}
gold_by_q, decomp_by_q = {}, {}
for r in json.load(open(os.path.join(a.dataset_dir, "musique.json"))):
    gold_by_q[r["question"]] = {key2id[(p["title"], p["paragraph_text"])] for p in r["paragraphs"] if p.get("is_supporting") and (p["title"], p["paragraph_text"]) in key2id}
    decomp_by_q[r["question"]] = [d["question"] for d in r.get("question_decomposition", [])]
if a.decomp and a.decomp != "oracle": decomp_by_q.update(json.load(open(a.decomp)))
rows = [json.loads(l) for l in open(a.details)]
if a.limit: rows = rows[:a.limit]
model, tok = load(a.model_path, adapter_path=a.adapter_path or None)
try: mx.set_cache_limit(2 << 30)   # bound MLX's buffer cache: full-vocab logits would otherwise pin tens of GB
except Exception: pass
_tm = model.language_model if hasattr(model, "language_model") else model
_inner = _tm.model
_head = (lambda h: _inner.embed_tokens.as_linear(h)) if getattr(_tm.args, "tie_word_embeddings", False) else _tm.lm_head
YES = tok.encode("yes", add_special_tokens=False)[0]; NO = tok.encode("no", add_special_tokens=False)[0]
PREFIX = "<|im_start|>system\n" + a.system + "<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def enc(q, doc):
    ids = tok.encode(PREFIX + f"<Instruct>: {a.instruct}\n<Query>: {q}\n<Document>: {doc}" + SUFFIX, add_special_tokens=False)
    if len(ids) >= a.max_len: ids = ids[: a.max_len - 40] + ids[-39:]
    return ids

def score(seqs):
    out = []
    for i in range(0, len(seqs), a.batch):
        chunk = seqs[i:i + a.batch]; L = max(len(s) for s in chunk)
        arr = mx.array([s + [0] * (L - len(s)) for s in chunk]); last = mx.array([len(s) - 1 for s in chunk])
        hidden = _inner(arr)[mx.arange(len(chunk)), last]        # [B, H] hidden states at each prompt's last token
        logits = _head(hidden)                                     # [B, V] only where needed (not B×L×V)
        two = mx.stack([logits[:, NO], logits[:, YES]], axis=1).astype(mx.float32)
        p = mx.softmax(two, axis=1)[:, 1]; mx.eval(p); out += p.tolist()
        try: mx.clear_cache()
        except Exception: pass
    return out

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg) / (len(pos) * len(neg))

res = []
for row in rows:
    q = row["question"]; gold = gold_by_q.get(q, set()); qt = q
    if a.decomp:
        sq = decomp_by_q.get(q, [])
        if sq: qt = q + "\nSub-questions: " + " | ".join(f"{i}) {s}" for i, s in enumerate(sq, 1))
    ids = [int(x) + a.id_shift for x in next(t for t in row["trace"] if t.get("stage") == "bm25")["candidate_document_ids"]]; ids = [i for i in ids if 0 <= i < len(corpus)]
    t0 = time.time(); sc = dict(zip(ids, score([enc(qt, f"{corpus[i]['title']}\n{corpus[i]['text']}") for i in ids]))); lat = time.time() - t0
    ranked = sorted(sc, key=lambda k: -sc[k]); pos = [sc[k] for k in sc if k in gold]; neg = [sc[k] for k in sc if k not in gold]
    if a.dump:
        with open(a.dump, "a") as f: f.write(json.dumps({"question": q, "model": a.tag or a.model_path, "scores": {str(k): v for k, v in sc.items()}, "gold": sorted(gold)}) + "\n")
    top10 = ranked[:10]; hits = [k for k in top10 if k in gold]; first = next((i + 1 for i, k in enumerate(ranked) if k in gold), None); gold_in = gold & set(sc)
    res.append({"auc": auc(pos, neg), "p10": len(hits) / 10, "r10": (len(hits) / len(gold_in)) if gold_in else float("nan"), "all10": float(gold_in <= set(top10)) if gold_in else float("nan"),
                "mrr": (1 / first) if first else 0.0, "gold_in_cands": len(gold_in), "gold": len(gold), "lat": lat})
def m(k):
    v = [r[k] for r in res if r[k] == r[k]]; return st.mean(v) if v else float("nan")
tag = a.tag or os.path.basename(a.model_path.rstrip("/"))
line = f"{tag:44s} n={len(res):3d} AUC={m('auc'):.3f} P@10={m('p10'):.3f} R@10={m('r10'):.3f} AllGold@10={m('all10'):.3f} MRR={m('mrr'):.3f} lat={m('lat'):.1f}s goldInCands={m('gold_in_cands'):.2f}/{m('gold'):.2f}"
print(line)
with open(os.path.expanduser("~/models/evals/sag2/judge-bench.tsv"), "a") as f: f.write(time.strftime("%m-%d %H:%M") + "  " + line + "\n")
