#!/usr/bin/env python3
"""llav v5 data: rows from REAL SAG2 rounds on training-split questions (MuSiQue / HotpotQA / 2Wiki, 300 q each), taken from
Jev-1.14 traces: per round the exact candidate set SAG2 produced, the evidence it passed, Jev's score, and the gold label.
Same state grammar as train-v4 (parsed by train-rerank-listwise.py). Negatives are subsampled: all positives + top-K by Jev
score (hard) + R random. Extra keys (jev, ds, hop) are kept for a later soft-target objective.
usage: build-v5.py OUT.jsonl [--hard 30] [--rand 10] [--ev-chars 700] [--ev-max 8] [--seed 5]"""
import json, os, random, sqlite3, sys
OUT = sys.argv[1]; args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
HARD = int(args.get("--hard", 30)); RAND = int(args.get("--rand", 10)); EVC = int(args.get("--ev-chars", 700)); EVM = int(args.get("--ev-max", 8)); random.seed(int(args.get("--seed", 5)))
T = os.path.expanduser("~/models/evals/sag2/train-sag2")
Q = {"verdict": {"type": "noul", "instructions": "Does the candidate passage itself state a fact that is necessary to answer the original question, including a fact that identifies an intermediate entity in the answer chain? Use the previously retrieved evidence only to connect facts. Topical association or background alone does not count. Use no outside knowledge.",
                 "criteria": {"true": "The candidate passage supplies a necessary fact (or a necessary intermediate identification) for answering the original question.", "false": "The candidate passage is only related background, concerns the wrong entity or relation, or the needed fact is only in other passages."}}}
def gold_docs(ds, s):
    if ds == "musique": return [f"{p['title']}\n{p['paragraph_text']}" for p in s["paragraphs"] if p["is_supporting"]]
    titles = {f[0] for f in s["supporting_facts"]}; sep = "" if ds == "hotpotqa" else " "
    return [f"{t}\n{sep.join(sents)}" for t, sents in s["context"] if t in titles]
n_rows = 0; stats = {}
with open(OUT, "w") as out:
    for ds in ("musique", "hotpotqa", "2wikimultihopqa"):
        corpus = json.load(open(f"{T}/dataset/{ds}_corpus.json")); key2id = {f"{c['title'].strip()}\n{c['text']}": i for i, c in enumerate(corpus)}
        # trace ids are CHUNK ids of the SQLite index (docs may be split into several chunks); chunk -> corpus index via source_id, text = chunk content (what the judge saw)
        db = sqlite3.connect(f"{T}/cache/{ds}.sqlite"); chunk = {cid: (int(sid), title, content) for cid, sid, title, content in db.execute("select id, source_id, title, content from chunks")}
        samples = {s["question"].strip(): s for s in json.load(open(f"{T}/dataset/{ds}.json"))}
        rows = [json.loads(l) for l in open(f"{T}/output/jev-{ds}-300/{ds}/details.jsonl")]
        st = {"q": 0, "rounds": 0, "pos": 0, "neg": 0, "gold_missing": 0}
        for r in rows:
            q = r["question"].strip(); s = samples.get(q)
            if not s: continue
            gold = {key2id[g] for g in gold_docs(ds, s) if g in key2id}; st["gold_missing"] += len(gold_docs(ds, s)) - len(gold); st["q"] += 1
            for t in r["trace"]:
                if t.get("stage") != "jev": continue
                st["rounds"] += 1; h = t["hop"]
                sc = {int(x["id"]): float(x["score"]) for x in t["scores"] if int(x["id"]) in chunk}
                ev = [int(x) for x in (t.get("evidence_ids") or []) if int(x) in chunk][:EVM]
                isg = lambda c: chunk[c][0] in gold
                pos = [c for c in sc if isg(c)]; neg = [c for c in sc if not isg(c)]
                hard = sorted(neg, key=lambda c: -sc[c])[:HARD]; rest = [c for c in neg if c not in set(hard)]; chosen = pos + hard + random.sample(rest, min(RAND, len(rest)))
                evblock = "".join(f"[{i}] {chunk[c][1].strip()}: {chunk[c][2][:EVC]}\n" for i, c in enumerate(ev, 1))
                for c in chosen:
                    state = f"Original question: {q}\n" + (f"Previously retrieved evidence (context only; may be wrong):\n{evblock}" if ev else "") + f"Candidate title: {chunk[c][1].strip()}\nCandidate passage: {chunk[c][2]}"
                    out.write(json.dumps({"state": state, "questions": Q, "expected": {"verdict": isg(c)}, "jev": sc[c], "ds": ds, "hop": h, "qid": s.get("_id", s.get("id"))}, ensure_ascii=False) + "\n")
                    n_rows += 1; st["pos" if isg(c) else "neg"] += 1
        stats[ds] = st; print(ds, st, flush=True)
print("rows", n_rows)
