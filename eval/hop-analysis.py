#!/usr/bin/env python3
"""Where does a judge lose? Per-hop analysis on MuSiQue: for each gold passage we know which hop of the decomposition
it supports (paragraph_support_idx); the last hop's passage holds the final answer, earlier hops are "bridge" passages
that only identify an intermediate entity. Reports, per model: mean rank and recall@10 of bridge vs final-hop gold,
and the fraction of questions where ALL gold passages are in the top 10.
Inputs: dump JSONL files written by the benches (--dump), or round-0 entries of the proxy log (--proxy-log + --filter).
usage: hop-analysis.py [--dump f.jsonl ...] [--proxy-log proxy-log.jsonl --filter typesafe] [--k 10]"""
import argparse, json, os, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("--dump", nargs="*", default=[]); ap.add_argument("--proxy-log", default=""); ap.add_argument("--filter", default="")
ap.add_argument("--k", type=int, default=10); ap.add_argument("--shift", type=int, default=0, help="-1 for end-to-end proxy logs (1-based chunk ids)"); ap.add_argument("--dataset-dir", default=os.path.expanduser("~/models/evals/sag2/bench/SAG-Benchmark/sag2/dataset"))
a = ap.parse_args()
corpus = json.load(open(os.path.join(a.dataset_dir, "musique_corpus.json")))
key2id = {(c["title"], c["text"]): i for i, c in enumerate(corpus)}
hop_of = {}  # question -> {corpus_id: (hop_index, n_hops)}
for r in json.load(open(os.path.join(a.dataset_dir, "musique.json"))):
    dec = r.get("question_decomposition", []); n = len(dec); m = {}
    for h, d in enumerate(dec):
        p = r["paragraphs"][d["paragraph_support_idx"]] if d.get("paragraph_support_idx") is not None else None
        if p and (p["title"], p["paragraph_text"]) in key2id: m[key2id[(p["title"], p["paragraph_text"])]] = (h, n)
    hop_of[r["question"]] = m

runs = {}  # model -> list of (question, scores)
for f in a.dump:
    for line in open(f):
        e = json.loads(line); runs.setdefault(e.get("model", f), []).append((e["question"], {int(k): v for k, v in e["scores"].items()}))
if a.proxy_log:
    for line in open(a.proxy_log):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("prev_evidence", 0) != 0 or (a.filter and a.filter not in e.get("model", "")) or e.get("question") not in hop_of: continue
        sc = {int(k.split("_")[-1]) + a.shift: v for k, v in (e.get("nouls") or {}).items() if v is not None}
        if sc: runs.setdefault(e["model"], []).append((e["question"], sc))

print(f"{'model':46s}   n  bridge:rank/R@k   final:rank/R@k   allgold@k  gold_missing_from_cands")
for model, items in runs.items():
    seen = set(); br = []; fr = []; allg = []; miss = 0; tot = 0
    for q, sc in items:
        if q in seen: continue
        seen.add(q); hops = hop_of.get(q, {})
        if not hops: continue
        ranked = sorted(sc, key=lambda c: -sc[c]); pos = {c: i + 1 for i, c in enumerate(ranked)}
        inc = [c for c in hops if c in pos]; miss += len(hops) - len(inc); tot += len(hops)
        for c in inc:
            h, n = hops[c]; (fr if h == n - 1 else br).append(pos[c])
        allg.append(float(all(pos[c] <= a.k for c in inc)) if inc else 0.0)
    def rk(v): return f"{st.mean(v):5.1f}/{st.mean([x <= a.k for x in v]):.2f}" if v else "   -  "
    print(f"{model[:46]:46s} {len(seen):3d}  {rk(br):>16s} {rk(fr):>16s}   {st.mean(allg) if allg else float('nan'):.3f}      {miss}/{tot}")
