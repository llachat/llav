#!/usr/bin/env python3
"""Replay the evidence-conditioned rounds (hop 1, hop 2) of a recorded SAG2/Jev run against an MLX reranker.
The Jev trace stores, per hop, the exact candidate set Jev scored, Jev's scores, and the evidence ids that were passed as
previously_retrieved_evidence. We rebuild the same request (same evidence, same candidates, our server's query format) and
score it with our model, so judge quality in rounds 1–2 is measured in isolation — no end-to-end run needed.
Ids in traces are 1-based chunk rowids (corpus index = id - 1).
usage: round-bench.py --details D.jsonl [--model-path DIR [--adapter-path A]] [--evidence-style v1|v4] [--hops 1,2] [--limit N] [--tag T] [--dump f.jsonl] [--jev-only]"""
import argparse, json, os, statistics as st, time
ap = argparse.ArgumentParser()
ap.add_argument("--details", required=True); ap.add_argument("--model-path", default=""); ap.add_argument("--adapter-path", default="")
ap.add_argument("--evidence-style", default="v1", choices=["v1", "v4"]); ap.add_argument("--evidence-chars", type=int, default=300)
ap.add_argument("--hops", default="1,2"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--dump", default="")
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--max-len", type=int, default=2048); ap.add_argument("--jev-only", action="store_true")
ap.add_argument("--instruct", default="Given a multi-hop question, judge whether the passage states a fact that is necessary to answer it (including a fact that identifies an intermediate entity in the answer chain).")
ap.add_argument("--system", default="Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".")
ap.add_argument("--dataset-dir", default=os.path.expanduser("~/models/evals/sag2/bench/SAG-Benchmark/sag2/dataset"))
a = ap.parse_args(); hops = [int(h) for h in a.hops.split(",")]
corpus = json.load(open(os.path.join(a.dataset_dir, "musique_corpus.json"))); key2id = {(c["title"], c["text"]): i for i, c in enumerate(corpus)}
gold_by_q = {}
for r in json.load(open(os.path.join(a.dataset_dir, "musique.json"))):
    gold_by_q[r["question"]] = {key2id[(p["title"], p["paragraph_text"])] for p in r["paragraphs"] if p.get("is_supporting") and (p["title"], p["paragraph_text"]) in key2id}
rows = [json.loads(l) for l in open(a.details)]
if a.limit: rows = rows[:a.limit]

def build_query(q, ev_ids):
    parts = []
    for i, cid in enumerate(ev_ids[:(8 if a.evidence_style == "v4" else 10)], 1):
        c = corpus[cid]; tag = f"[{i}] " if a.evidence_style == "v4" else ""
        parts.append(f"{tag}{c['title']}: {c['text'][:a.evidence_chars]}")
    return q + ("\nAlready retrieved evidence (context only): " + " || ".join(parts) if parts else "")

score = None
if not a.jev_only:
    import mlx.core as mx
    import mlx_lm.utils as _U; _U.MODEL_REMAPPING["qwen3_5_text"] = "qwen3_5"
    from mlx_lm.utils import load
    model, tok = load(a.model_path, adapter_path=a.adapter_path or None)
    try: mx.set_cache_limit(2 << 30)
    except Exception: pass
    _tm = model.language_model if hasattr(model, "language_model") else model; _inner = _tm.model
    _head = (lambda h: _inner.embed_tokens.as_linear(h)) if getattr(_tm.args, "tie_word_embeddings", False) else _tm.lm_head
    YES = tok.encode("yes", add_special_tokens=False)[0]; NO = tok.encode("no", add_special_tokens=False)[0]
    PREFIX = "<|im_start|>system\n" + a.system + "<|im_end|>\n<|im_start|>user\n"; SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    def enc(q, doc):
        ids = tok.encode(PREFIX + f"<Instruct>: {a.instruct}\n<Query>: {q}\n<Document>: {doc}" + SUFFIX, add_special_tokens=False)
        if len(ids) >= a.max_len: ids = ids[: a.max_len - 40] + ids[-39:]
        return ids
    def score(q, cids):
        seqs = [enc(q, f"{corpus[i]['title']}\n{corpus[i]['text']}") for i in cids]; out = []
        for i in range(0, len(seqs), a.batch):
            chunk = seqs[i:i + a.batch]; L = max(len(s) for s in chunk)
            arr = mx.array([s + [0] * (L - len(s)) for s in chunk]); last = mx.array([len(s) - 1 for s in chunk])
            lg = _head(_inner(arr)[mx.arange(len(chunk)), last]); p = mx.softmax(mx.stack([lg[:, NO], lg[:, YES]], axis=1).astype(mx.float32), axis=1)[:, 1]; mx.eval(p); out += p.tolist()
            try: mx.clear_cache()
            except Exception: pass
        return out

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg) / (len(pos) * len(neg))

def metrics(sc, gold, excl):
    """sc: {corpus_id: score}; excl: evidence already retrieved (ignored for 'new gold' metrics)"""
    ranked = sorted(sc, key=lambda k: -sc[k]); pos = [sc[k] for k in sc if k in gold]; neg = [sc[k] for k in sc if k not in gold]
    new_gold = {g for g in gold if g in sc and g not in excl}; new_ranked = [k for k in ranked if k not in excl]
    r10 = (len([k for k in ranked[:10] if k in gold]) / len(gold & set(sc))) if gold & set(sc) else float("nan")
    newr10 = (len([k for k in new_ranked[:10] if k in new_gold]) / len(new_gold)) if new_gold else float("nan")
    return {"auc": auc(pos, neg), "r10": r10, "new_r10": newr10, "n_new_gold": len(new_gold), "n_gold_in": len(gold & set(sc)), "n": len(sc)}

per_hop = {h: {"ours": [], "jev": []} for h in hops}; lat = []
for row in rows:
    q = row["question"]; gold = gold_by_q.get(q, set())
    for t in row["trace"]:
        if t.get("stage") != "jev" or t.get("hop") not in hops: continue
        h = t["hop"]; jev_sc = {int(s["id"]) - 1: float(s["score"]) for s in t["scores"] if 0 <= int(s["id"]) - 1 < len(corpus)}
        ev = [int(x) - 1 for x in t.get("evidence_ids") or [] if 0 <= int(x) - 1 < len(corpus)]; excl = set(ev)
        per_hop[h]["jev"].append(metrics(jev_sc, gold, excl))
        if score is not None:
            cids = list(jev_sc); t0 = time.time(); ours = dict(zip(cids, score(build_query(q, ev), cids))); lat.append(time.time() - t0)
            per_hop[h]["ours"].append(metrics(ours, gold, excl))
            if a.dump:
                with open(a.dump, "a") as f: f.write(json.dumps({"question": q, "hop": h, "model": a.tag, "evidence": ev, "scores": {str(k): v for k, v in ours.items()}, "jev": {str(k): v for k, v in jev_sc.items()}, "gold": sorted(gold)}) + "\n")

def m(lst, k):
    v = [r[k] for r in lst if r[k] == r[k]]; return st.mean(v) if v else float("nan")
lines = []
for h in hops:
    for who in (["jev"] if score is None else ["jev", "ours"]):
        L = per_hop[h][who]; name = ("Jev(trace)" if who == "jev" else (a.tag or os.path.basename(a.model_path)))
        lines.append(f"hop{h} {name:36s} n={len(L):3d} AUC={m(L,'auc'):.3f} R@10={m(L,'r10'):.3f} newGoldR@10={m(L,'new_r10'):.3f} cands={m(L,'n'):.0f} goldIn={m(L,'n_gold_in'):.2f} newGold={m(L,'n_new_gold'):.2f}" + (f" lat={st.mean(lat):.1f}s" if who == "ours" and lat else ""))
print("\n".join(lines))
with open(os.path.expanduser("~/models/evals/sag2/round-bench.tsv"), "a") as f:
    f.write(time.strftime("%m-%d %H:%M") + f"  [{os.path.basename(os.path.dirname(os.path.dirname(a.details)))} style={a.evidence_style}]\n" + "\n".join("    " + l for l in lines) + "\n")
