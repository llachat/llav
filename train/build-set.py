#!/usr/bin/env python3
"""Set-level judge training data (bet 1, llav-set): one row = one question + a SET of candidate passages with gold labels.
The model will see all candidates in one sequence and emit a verdict per candidate (chain-consistent, no evidence lock-in).

Sources
  traces : Jev traces on the training-split questions (3 datasets, 2,700 rounds): per round the set = Jev's top-K by score
           ∪ every gold passage present in the round's candidate pool (so the set always contains the answerable chain).
  native : MuSiQue train (20 paragraphs, 2–4 gold), HotpotQA distractor train (10 paragraphs, 2 gold), 2Wiki train (10, 2–4).
Rows: {"q", "items": [{"title","text","label"}], "src", "hop", "n_gold"}. Order is shuffled at training time, not here.
usage: build-set.py OUT.jsonl [--topk 30] [--native-per-ds 6000] [--seed 7]"""
import json, os, random, sqlite3, sys
import pyarrow.parquet as pq
OUT = sys.argv[1]; args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
TOPK = int(args.get("--topk", 30)); NPER = int(args.get("--native-per-ds", 6000)); random.seed(int(args.get("--seed", 7)))
T = os.path.expanduser("~/models/evals/sag2/train-sag2"); DATA = os.path.expanduser("~/models/evals/sag2/data"); RAW = f"{DATA}/raw"
EVAL = os.path.expanduser("~/models/evals/sag2/bench/SAG-Benchmark/sag2/dataset")
eval_q = set()
for n in ("musique", "hotpotqa", "2wikimultihopqa"):
    eval_q |= {q.strip() for q in json.load(open(f"{DATA}/eval-ids-{n}.json"))["questions"]}

def gold_docs(ds, s):
    if ds == "musique": return [f"{p['title']}\n{p['paragraph_text']}" for p in s["paragraphs"] if p["is_supporting"]]
    titles = {f[0] for f in s["supporting_facts"]}; sep = "" if ds == "hotpotqa" else " "
    return [f"{t}\n{sep.join(sents)}" for t, sents in s["context"] if t in titles]

n = 0; stats = {}
with open(OUT, "w") as out:
    # ---- (i) trace sets
    for ds in ("musique", "hotpotqa", "2wikimultihopqa"):
        corpus = json.load(open(f"{T}/dataset/{ds}_corpus.json")); key2id = {f"{c['title'].strip()}\n{c['text']}": i for i, c in enumerate(corpus)}
        db = sqlite3.connect(f"{T}/cache/{ds}.sqlite"); chunk = {cid: (int(sid), title, content) for cid, sid, title, content in db.execute("select id, source_id, title, content from chunks")}
        samples = {s["question"].strip(): s for s in json.load(open(f"{T}/dataset/{ds}.json"))}
        st = {"rows": 0, "pos": 0, "items": 0}
        for r in (json.loads(l) for l in open(f"{T}/output/jev-{ds}-300/{ds}/details.jsonl")):
            q = r["question"].strip(); s = samples.get(q)
            if not s: continue
            gold = {key2id[g] for g in gold_docs(ds, s) if g in key2id}
            for t in r["trace"]:
                if t.get("stage") != "jev": continue
                sc = {int(x["id"]): float(x["score"]) for x in t["scores"] if int(x["id"]) in chunk}
                top = sorted(sc, key=lambda c: -sc[c])[:TOPK]; pool_gold = [c for c in sc if chunk[c][0] in gold]
                ids = list(dict.fromkeys(top + pool_gold))
                if not pool_gold or len(ids) < 4: continue
                items = [{"title": chunk[c][1].strip(), "text": chunk[c][2], "label": int(chunk[c][0] in gold)} for c in ids]
                out.write(json.dumps({"q": q, "items": items, "src": f"trace-{ds}", "hop": t["hop"], "n_gold": len(pool_gold)}, ensure_ascii=False) + "\n")
                n += 1; st["rows"] += 1; st["pos"] += len(pool_gold); st["items"] += len(items)
        stats[f"trace-{ds}"] = st; print(f"trace-{ds}", st, flush=True)

    # ---- (ii) native sets
    def emit(ds, q, items):
        global n
        if q.strip() in eval_q or not any(i["label"] for i in items) or all(i["label"] for i in items): return
        out.write(json.dumps({"q": q.strip(), "items": items, "src": f"native-{ds}", "hop": -1, "n_gold": sum(i["label"] for i in items)}, ensure_ascii=False) + "\n"); n += 1
    rows = [json.loads(l) for l in open(f"{DATA}/musique_ans_v1.0_train.jsonl")]; random.shuffle(rows)
    for s in rows[:NPER]:
        emit("musique", s["question"], [{"title": p["title"], "text": p["paragraph_text"], "label": int(p["is_supporting"])} for p in s["paragraphs"]])
    print("native-musique done", flush=True)
    hot = []
    for f in sorted(x for x in os.listdir(RAW) if x.startswith("hotpotqa_")):
        hot += pq.read_table(f"{RAW}/{f}").to_pylist()
    random.shuffle(hot)
    for r in hot[:NPER]:
        titles = set(r["supporting_facts"]["title"])
        emit("hotpotqa", r["question"], [{"title": t, "text": "".join(sents), "label": int(t in titles)} for t, sents in zip(r["context"]["title"], r["context"]["sentences"])])
    print("native-hotpotqa done", flush=True)
    wk = pq.read_table(f"{RAW}/xanhho_2WikiMultihopQA__train.parquet").to_pylist(); random.shuffle(wk)
    for r in wk[:NPER]:
        ctx = json.loads(r["context"]); titles = {f[0] for f in json.loads(r["supporting_facts"])}
        emit("2wikimultihopqa", r["question"], [{"title": t, "text": " ".join(sents), "label": int(t in titles)} for t, sents in ctx])
    print("native-2wiki done", flush=True)
print("total rows", n)
