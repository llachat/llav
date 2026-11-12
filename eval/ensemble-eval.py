#!/usr/bin/env python3
"""Ensemble judges offline from their per-question score dumps (and/or proxy-log round-0 entries).
Per question: normalise each judge's 100 candidate scores (rank → [0,1], or z-score, or raw), average with weights,
then compute AUC / R@10 / AllGold@10 / MRR exactly like judge-bench. Also prints each member alone on the common question set.

usage: ensemble-eval.py --member TAG=dump.jsonl --member TAG=proxylog:FILTER ... [--norm rank|z|raw] [--weights 1,1,...] [--k 10]
"""
import argparse, json, os, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("--member", action="append", required=True); ap.add_argument("--norm", default="rank", choices=["rank", "z", "raw"])
ap.add_argument("--weights", default=""); ap.add_argument("--k", type=int, default=10)
ap.add_argument("--dataset-dir", default=os.path.expanduser("~/models/evals/sag2/bench/SAG-Benchmark/sag2/dataset")); ap.add_argument("--proxy-log", default=os.path.expanduser("~/models/evals/sag2/proxy-log.jsonl"))
a = ap.parse_args()
corpus = json.load(open(os.path.join(a.dataset_dir, "musique_corpus.json"))); key2id = {(c["title"], c["text"]): i for i, c in enumerate(corpus)}
gold = {}
for r in json.load(open(os.path.join(a.dataset_dir, "musique.json"))):
    gold[r["question"]] = {key2id[(p["title"], p["paragraph_text"])] for p in r["paragraphs"] if p.get("is_supporting") and (p["title"], p["paragraph_text"]) in key2id}

def load_member(spec):
    tag, _, src = spec.partition("="); out = {}
    if src.startswith("proxylog:"):
        filt = src.split(":", 1)[1]
        for line in open(a.proxy_log):
            try: e = json.loads(line)
            except Exception: continue
            if filt in e.get("model", "") and e.get("prev_evidence", 0) == 0 and e.get("question") in gold and e["question"] not in out:
                sc = {int(k.split("_")[-1]): v for k, v in (e.get("nouls") or {}).items() if v is not None}
                if len(sc) >= 50: out[e["question"]] = sc
    else:
        for line in open(os.path.expanduser(src)):
            e = json.loads(line)
            if e["question"] not in out: out[e["question"]] = {int(k): v for k, v in e["scores"].items()}
    return tag, out

members = [load_member(m) for m in a.member]
common = set.intersection(*(set(m[1]) for m in members))
w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(members)

def norm(sc):
    ids = list(sc); vals = [sc[i] for i in ids]
    if a.norm == "raw": return dict(zip(ids, vals))
    if a.norm == "z":
        mu = st.mean(vals); sd = st.pstdev(vals) or 1.0; return {i: (v - mu) / sd for i, v in zip(ids, vals)}
    order = sorted(ids, key=lambda i: sc[i]); n = len(order)
    return {i: r / max(n - 1, 1) for r, i in enumerate(order)}

def metrics(scores_by_q):
    aucs, r10, all10, mrr = [], [], [], []
    for q, sc in scores_by_q.items():
        g = gold[q] & set(sc)
        if not g: continue
        pos = [sc[i] for i in sc if i in g]; neg = [sc[i] for i in sc if i not in g]
        aucs.append(sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg) / (len(pos) * len(neg)))
        ranked = sorted(sc, key=lambda i: -sc[i]); top = ranked[:a.k]; hits = [i for i in top if i in g]
        r10.append(len(hits) / len(g)); all10.append(float(g <= set(top))); first = next((r + 1 for r, i in enumerate(ranked) if i in g), None); mrr.append(1 / first if first else 0)
    return f"AUC={st.mean(aucs):.3f} R@{a.k}={st.mean(r10):.3f} AllGold@{a.k}={st.mean(all10):.3f} MRR={st.mean(mrr):.3f} (n={len(aucs)})"

print(f"common questions: {len(common)}  norm={a.norm}")
for tag, m in members:
    print(f"  {tag:40s} {metrics({q: m[q] for q in common})}")
ens = {}
for q in common:
    acc = {}
    for (tag, m), wt in zip(members, w):
        for i, v in norm(m[q]).items(): acc[i] = acc.get(i, 0.0) + wt * v
    ens[q] = acc
print(f"  {'ENSEMBLE(' + '+'.join(t for t, _ in members) + ')':40s} {metrics(ens)}")
