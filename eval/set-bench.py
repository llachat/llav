#!/usr/bin/env python3
"""Evaluate a set-level judge (train-set-judge.py adapters) on recorded SAG2 rounds: for each question/hop take the round's
candidate pool, pre-select the top-K by a prefilter score (default: the pool's own recorded judge scores, i.e. Jev's — the
realistic setting is "0.6B prefilter → set model"), score the K jointly, and rank the pool (K by the set model, the rest keep
prefilter order below). Reports within-pool R@10 / AllGold@10 / AUC(K) and the same for the prefilter alone.
usage: set-bench.py --model-path DIR --adapter-path A --details D.jsonl [--hops 2] [--topk 30] [--limit N] [--tag T] [--dump f]
       [--prefilter jev|dump:FILE]   (dump = per-question scores from round-bench --dump, keyed by hop)"""
import argparse, json, os, random, statistics as st, time
import mlx.core as mx
from mlx_lm.utils import load
ap = argparse.ArgumentParser()
ap.add_argument("--model-path", required=True); ap.add_argument("--adapter-path", default=""); ap.add_argument("--details", required=True)
ap.add_argument("--hops", default="2"); ap.add_argument("--topk", type=int, default=30); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tag", default="llav-set")
ap.add_argument("--dump", default=""); ap.add_argument("--prefilter", default="jev"); ap.add_argument("--doc-tokens", type=int, default=110); ap.add_argument("--max-seq-length", type=int, default=5120)
ap.add_argument("--shuffles", type=int, default=1, help="average over N random orders (order-robustness)"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dataset-dir", default=os.path.expanduser("~/models/evals/sag2/bench/SAG-Benchmark/sag2/dataset"))
ap.add_argument("--instruct", default="Given a multi-hop question and a set of candidate passages, decide for EACH candidate whether it is necessary evidence in a fully supported answer chain (a bridge passage that identifies an intermediate entity counts). Candidates can only be judged together: a passage is necessary only if the chain it belongs to is consistent with the other selected passages.")
a = ap.parse_args(); random.seed(a.seed); hops = [int(h) for h in a.hops.split(",")]
corpus = json.load(open(os.path.join(a.dataset_dir, "musique_corpus.json"))); key2id = {(c["title"], c["text"]): i for i, c in enumerate(corpus)}
gold_by_q = {}
for r in json.load(open(os.path.join(a.dataset_dir, "musique.json"))):
    gold_by_q[r["question"]] = {key2id[(p["title"], p["paragraph_text"])] for p in r["paragraphs"] if p.get("is_supporting") and (p["title"], p["paragraph_text"]) in key2id}
pre_dump = {}
if a.prefilter.startswith("dump:"):
    for l in open(a.prefilter[5:]):
        e = json.loads(l); pre_dump[(e["question"], e.get("hop"))] = {int(k): v for k, v in e["scores"].items()}
rows = [json.loads(l) for l in open(a.details)]
if a.limit: rows = rows[: a.limit]
model, tok = load(a.model_path, adapter_path=a.adapter_path or None)
try: mx.set_cache_limit(4 << 30)
except Exception: pass
tm = model.language_model if hasattr(model, "language_model") else model
head = (lambda h: tm.model.embed_tokens.as_linear(h)) if getattr(tm.args, "tie_word_embeddings", False) else tm.lm_head
YES = tok.encode("yes", add_special_tokens=False)[0]; NO = tok.encode("no", add_special_tokens=False)[0]
SYSTEM = "Judge whether each candidate Document meets the requirements based on the Query and the Instruct provided. For every verdict slot the answer can only be \"yes\" or \"no\"."
PRE = tok.encode("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n" + f"<Instruct>: {a.instruct}\n<Query>: ", add_special_tokens=False)
MID = tok.encode("\n<Candidates>:\n", add_special_tokens=False); SUF = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nVerdicts:", add_special_tokens=False)
SLOT = [tok.encode(f" [{k}]", add_special_tokens=False) for k in range(0, a.topk + 1)]; HEAD = [tok.encode(f"[{k}] ", add_special_tokens=False) for k in range(0, a.topk + 1)]

def score_set(q, docs):
    ids = PRE + tok.encode(q, add_special_tokens=False) + MID
    budget = a.max_seq_length - len(ids) - len(SUF) - sum(len(s) for s in SLOT[1:len(docs) + 1]) - 8; per = max(40, min(a.doc_tokens, budget // max(len(docs), 1) - 6))
    for k, (title, text) in enumerate(docs, 1): ids += HEAD[k] + tok.encode(f"{title}: {text}", add_special_tokens=False)[:per] + tok.encode("\n", add_special_tokens=False)
    ids += SUF; pos = []
    for k in range(1, len(docs) + 1): ids += SLOT[k]; pos.append(len(ids) - 1)
    h = tm.model(mx.array([ids]))[0][mx.array(pos)]; lg = head(h); m = (lg[:, YES] - lg[:, NO]).astype(mx.float32); mx.eval(m)
    try: mx.clear_cache()
    except Exception: pass
    return m.tolist()

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg) / (len(pos) * len(neg))

res = {h: {"set": [], "pre": []} for h in hops}; lat = []
for row in rows:
    q = row["question"]; gold = gold_by_q.get(q, set())
    for t in row["trace"]:
        if t.get("stage") != "jev" or t.get("hop") not in hops: continue
        h = t["hop"]; jev = {int(x["id"]) - 1: float(x["score"]) for x in t["scores"] if 0 <= int(x["id"]) - 1 < len(corpus)}
        pre = pre_dump.get((q, h), jev) if pre_dump else jev
        pre = {c: pre.get(c, -1.0) for c in jev}
        top = sorted(pre, key=lambda c: -pre[c])[: a.topk]
        gin = gold & set(jev)
        if not gin: continue
        t0 = time.time(); acc = {c: 0.0 for c in top}
        for s in range(a.shuffles):
            order = top[:]; random.shuffle(order)
            ms = score_set(q, [(corpus[c]["title"], corpus[c]["text"]) for c in order])
            for c, m in zip(order, ms): acc[c] += m / a.shuffles
        lat.append(time.time() - t0)
        lo = min(acc.values()) - 1.0; final = dict(acc)
        for c in jev:
            if c not in final: final[c] = lo - (1.0 - pre[c])          # non-selected stay below the set, in prefilter order
        def metrics(sc):
            ranked = sorted(sc, key=lambda c: -sc[c]); top10 = ranked[:10]
            return {"r10": len([c for c in top10 if c in gin]) / len(gin), "all10": float(gin <= set(top10)), "auc": auc([sc[c] for c in sc if c in gold], [sc[c] for c in sc if c not in gold]), "gold_in_topk": len(gin & set(top)) / len(gin)}
        res[h]["set"].append(metrics(final)); res[h]["pre"].append(metrics(pre))
        if a.dump:
            with open(a.dump, "a") as f: f.write(json.dumps({"question": q, "hop": h, "model": a.tag, "scores": {str(c): v for c, v in final.items()}, "gold": sorted(gold)}) + "\n")
def m(L, k):
    v = [x[k] for x in L if x[k] == x[k]]; return st.mean(v) if v else float("nan")
lines = []
for h in hops:
    for who in ("pre", "set"):
        L = res[h][who]; name = ("prefilter(" + a.prefilter + ")") if who == "pre" else a.tag
        lines.append(f"hop{h} {name:34s} n={len(L):3d} poolR@10={m(L,'r10'):.3f} AllGold@10={m(L,'all10'):.3f} AUC={m(L,'auc'):.3f} goldInTop{a.topk}={m(L,'gold_in_topk'):.3f}" + (f" lat={st.mean(lat):.1f}s" if who == "set" and lat else ""))
print("\n".join(lines))
with open(os.path.expanduser("~/models/evals/sag2/set-bench.tsv"), "a") as f: f.write(time.strftime("%m-%d %H:%M") + f"  [{os.path.basename(os.path.dirname(os.path.dirname(a.details)))}]\n" + "\n".join("    " + l for l in lines) + "\n")
