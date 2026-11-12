#!/usr/bin/env python3
"""Zero-shot zerank-2 (ZeroEntropy, Qwen3-4B base, Elo-trained reranker) on the judge benchmark, via sentence-transformers
CrossEncoder on MPS (the model ships its own chat template + logit-score module, so we use their official path).
usage: zerank-bench.py --model-path DIR --details DETAILS.jsonl [--limit N] [--batch 8] [--tag T] [--dump f.jsonl] [--id-shift 0]"""
import argparse, json, os, statistics as st, time, torch
from sentence_transformers import CrossEncoder
ap = argparse.ArgumentParser()
ap.add_argument("--model-path", required=True); ap.add_argument("--details", required=True); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--tag", default="zerank-2 zero-shot"); ap.add_argument("--dump", default=""); ap.add_argument("--id-shift", type=int, default=0)
ap.add_argument("--dataset-dir", default=os.path.expanduser("~/models/evals/sag2/bench/SAG-Benchmark/sag2/dataset"))
a = ap.parse_args()
corpus = json.load(open(os.path.join(a.dataset_dir, "musique_corpus.json"))); key2id = {(c["title"], c["text"]): i for i, c in enumerate(corpus)}
gold_by_q = {}
for r in json.load(open(os.path.join(a.dataset_dir, "musique.json"))):
    gold_by_q[r["question"]] = {key2id[(p["title"], p["paragraph_text"])] for p in r["paragraphs"] if p.get("is_supporting") and (p["title"], p["paragraph_text"]) in key2id}
rows = [json.loads(l) for l in open(a.details)]
if a.limit: rows = rows[:a.limit]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = CrossEncoder(a.model_path, device=dev, model_kwargs={"torch_dtype": torch.float16})

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg) / (len(pos) * len(neg))

res = []
for row in rows:
    q = row["question"]; gold = gold_by_q.get(q, set())
    ids = [int(x) + a.id_shift for x in next(t for t in row["trace"] if t.get("stage") == "bm25")["candidate_document_ids"]]; ids = [i for i in ids if 0 <= i < len(corpus)]
    t0 = time.time()
    scores = model.predict([(q, f"{corpus[i]['title']}\n{corpus[i]['text']}") for i in ids], batch_size=a.batch, convert_to_numpy=True)
    sc = {i: float(s) for i, s in zip(ids, scores)}; lat = time.time() - t0
    ranked = sorted(sc, key=lambda k: -sc[k]); pos = [sc[k] for k in sc if k in gold]; neg = [sc[k] for k in sc if k not in gold]
    top10 = ranked[:10]; hits = [k for k in top10 if k in gold]; first = next((i + 1 for i, k in enumerate(ranked) if k in gold), None); gold_in = gold & set(sc)
    res.append({"auc": auc(pos, neg), "p10": len(hits) / 10, "r10": (len(hits) / len(gold_in)) if gold_in else float("nan"), "all10": float(gold_in <= set(top10)) if gold_in else float("nan"), "mrr": (1 / first) if first else 0.0, "gold_in_cands": len(gold_in), "gold": len(gold), "lat": lat})
    if a.dump:
        with open(a.dump, "a") as f: f.write(json.dumps({"question": q, "model": a.tag, "scores": {str(k): v for k, v in sc.items()}, "gold": sorted(gold)}) + "\n")
def m(k):
    v = [r[k] for r in res if r[k] == r[k]]; return st.mean(v) if v else float("nan")
line = f"{a.tag:44s} n={len(res):3d} AUC={m('auc'):.3f} P@10={m('p10'):.3f} R@10={m('r10'):.3f} AllGold@10={m('all10'):.3f} MRR={m('mrr'):.3f} lat={m('lat'):.1f}s goldInCands={m('gold_in_cands'):.2f}/{m('gold'):.2f}"
print(line)
with open(os.path.expanduser("~/models/evals/sag2/judge-bench.tsv"), "a") as f: f.write(time.strftime("%m-%d %H:%M") + "  " + line + "\n")
