#!/usr/bin/env python3
"""Hard-negative training data for the SAG2 evidence judge, from the MuSiQue *training split*.

Why: in the real SAG2 pipeline the judge sees BM25 neighbours of the question and, in later rounds, passages pulled
in by terms from the seed passages. Those are much harder than the 18 distractors shipped with each question
(fine-tune v1: AUC 0.87 on round-0 candidates but 0.68 across expansion rounds). So we rebuild that distribution:
  corpus  = all paragraphs of the training split (deduplicated), in SQLite FTS5 (same idea as SAG2's index)
  round 0 = BM25 top-100 for the question; negatives sampled from ranks 1–20 and 21–100
  round 1 = BM25 top-40 for "question + terms of one gold passage" (expansion), gold passage placed in the state as
            previously retrieved evidence; negatives from that list
Rows use exactly the proxy's batch-mode state text.
usage: build-hard.py OUT.jsonl --musique-train PATH [--questions N] [--seed S]"""
import json, os, random, re, sqlite3, sys
OUT = sys.argv[1]; MT = sys.argv[sys.argv.index("--musique-train") + 1]
N = int(sys.argv[sys.argv.index("--questions") + 1]) if "--questions" in sys.argv else 6000
random.seed(int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 7)
DS = os.environ.get("SAG2_DATASET_DIR", "/Users/williamsing/models/evals/sag2/bench/SAG-Benchmark/sag2/dataset")
BATCH_QUESTION = ("Does the candidate passage itself state a fact that is necessary to answer the original question, "
                  "including a fact that identifies an intermediate entity in the answer chain? Use the previously retrieved "
                  "evidence only to connect facts. Topical association or background alone does not count. Use no outside knowledge.")
BATCH_CRITERIA = {"true": "The candidate passage supplies a necessary fact (or a necessary intermediate identification) for answering the original question.",
                  "false": "The candidate passage is only related background, concerns the wrong entity or relation, or the needed fact is only in other passages."}
Q = {"verdict": {"type": "noul", "instructions": BATCH_QUESTION, "criteria": BATCH_CRITERIA}}
STOP = set("the a an of in on at to for and or is was were are be by with from as that this which who whom whose what when where why how did does do has have had it its his her their he she they them than then into about after before between during over under".split())
test_ids = {r["id"] for r in json.load(open(os.path.join(DS, "musique.json")))}
rows_ = [json.loads(l) for l in open(MT)]; rows_ = [r for r in rows_ if r["id"] not in test_ids]; random.shuffle(rows_)

def state_text(question, title, passage, evidence):
    s = f"Original question: {question}\n"
    if evidence:
        s += "Previously retrieved evidence (context only; may be wrong):\n"
        for i, (t, p) in enumerate(evidence[:10], 1): s += f"[{i}] {t}: {p[:400]}\n"
    return s + f"Candidate title: {title}\nCandidate passage: {passage}"

def terms(text, limit=40):
    out = []
    for w in re.findall(r"[0-9A-Za-zÀ-ɏ一-鿿]+", text):
        lw = w.lower()
        if lw in STOP or len(lw) < 2: continue
        if lw not in out: out.append(lw)
    return out[:limit]

def fts_query(text, limit=40):
    ts = terms(text, limit)
    return " OR ".join('"' + t.replace('"', '') + '"' for t in ts) if ts else '""'

# corpus + index (in-memory)
db = sqlite3.connect(":memory:"); db.execute("CREATE VIRTUAL TABLE p USING fts5(title, body, tokenize='unicode61')")
key2id = {}; docs = []
for r in rows_:
    for pr in r["paragraphs"]:
        k = (pr["title"], pr["paragraph_text"])
        if k not in key2id: key2id[k] = len(docs); docs.append(k)
db.executemany("INSERT INTO p(rowid, title, body) VALUES (?,?,?)", [(i, t, b) for i, (t, b) in enumerate(docs)]); db.commit()
print(f"corpus {len(docs)} paragraphs from {len(rows_)} training questions", file=sys.stderr)

def search(text, k):
    try:
        cur = db.execute("SELECT rowid FROM p WHERE p MATCH ? ORDER BY bm25(p) LIMIT ?", (fts_query(text), k))
        return [row[0] for row in cur]
    except sqlite3.OperationalError:
        return []

out = []; stats = {"q": 0, "pos": 0, "neg0": 0, "neg1": 0, "gold_in_top100": 0}
for r in rows_[:N]:
    q = r["question"]; gold = [(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if p.get("is_supporting")]
    gold_ids = {key2id[g] for g in gold}
    top = search(q, 100); stats["q"] += 1; stats["gold_in_top100"] += len(gold_ids & set(top)) / max(len(gold_ids), 1)
    # positives: all gold (40% with the other gold as evidence)
    for t, p in gold:
        ev = [g for g in gold if g != (t, p)] if random.random() < 0.4 else []
        out.append({"state": state_text(q, t, p, ev), "questions": Q, "expected": {"verdict": True}}); stats["pos"] += 1
    neg_top = [i for i in top[:20] if i not in gold_ids]; neg_tail = [i for i in top[20:] if i not in gold_ids]
    for i in random.sample(neg_top, min(4, len(neg_top))) + random.sample(neg_tail, min(3, len(neg_tail))):
        t, p = docs[i]; ev = random.sample(gold, 1) if random.random() < 0.3 else []
        out.append({"state": state_text(q, t, p, ev), "questions": Q, "expected": {"verdict": False}}); stats["neg0"] += 1
    # expansion round: seed = one gold passage; query = question + seed terms; evidence = that seed
    seed = random.choice(gold); exp = search(q + " " + seed[0] + " " + seed[1][:300], 40)
    neg_exp = [i for i in exp if i not in gold_ids]
    for i in random.sample(neg_exp, min(3, len(neg_exp))):
        t, p = docs[i]
        out.append({"state": state_text(q, t, p, [seed]), "questions": Q, "expected": {"verdict": False}}); stats["neg1"] += 1
random.shuffle(out)
with open(OUT, "w") as f:
    for it in out: f.write(json.dumps(it, ensure_ascii=False) + "\n")
stats["gold_in_top100"] = round(stats["gold_in_top100"] / max(stats["q"], 1), 3)
print(json.dumps({"rows": len(out), **stats}), file=sys.stderr)
