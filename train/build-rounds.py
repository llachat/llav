#!/usr/bin/env python3
"""Round-aware training data: mimic SAG2 rounds 1–2, where the judge sees previously retrieved evidence (a mix of right and
wrong passages) and must judge expansion-round candidates. This is where llav-0.1 loses to Jev (AUC 0.91 vs 0.98).

For each MuSiQue-train question (not in the test set):
  round 0 : BM25 top-100 over the training-split corpus (SQLite FTS5)                              -> some rows w/o evidence
  evidence: gold passages found in top-100 (as if the judge got them) + 2–5 hard negatives from top-20 (judge mistakes)
  round 1 : expansion hits = BM25 top-40 for (question + terms of each evidence passage)            -> rows WITH evidence
  round 2 : expansion from (question + a round-1 gold/near miss)                                    -> rows WITH evidence
Labels = is_supporting. Evidence passages are kept at full length (cap --ev-chars each, --ev-max items) so the model can
actually use them; the proxy/llav-serve must pass the same format at inference.
State text (same grammar as the batch mode of the proxy):
  Original question: ...\nPreviously retrieved evidence (context only; may be wrong):\n[1] title: passage\n...\nCandidate title: ...\nCandidate passage: ...
usage: build-rounds.py OUT.jsonl --musique-train PATH --questions N [--ev-chars 700] [--ev-max 8] [--seed 11]"""
import json, os, random, re, sqlite3, sys
OUT = sys.argv[1]; MT = sys.argv[sys.argv.index("--musique-train") + 1]
N = int(sys.argv[sys.argv.index("--questions") + 1]) if "--questions" in sys.argv else 4000
EVC = int(sys.argv[sys.argv.index("--ev-chars") + 1]) if "--ev-chars" in sys.argv else 700
EVM = int(sys.argv[sys.argv.index("--ev-max") + 1]) if "--ev-max" in sys.argv else 8
random.seed(int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 11)
DS = os.environ.get("SAG2_DATASET_DIR", os.path.expanduser("~/models/evals/sag2/bench/SAG-Benchmark/sag2/dataset"))
Q = {"verdict": {"type": "noul", "instructions": "Does the candidate passage itself state a fact that is necessary to answer the original question, including a fact that identifies an intermediate entity in the answer chain? Use the previously retrieved evidence only to connect facts. Topical association or background alone does not count. Use no outside knowledge.",
                 "criteria": {"true": "The candidate passage supplies a necessary fact (or a necessary intermediate identification) for answering the original question.", "false": "The candidate passage is only related background, concerns the wrong entity or relation, or the needed fact is only in other passages."}}}
STOP = set("the a an of in on at to for and or is was were are be by with from as that this which who whom whose what when where why how did does do has have had it its his her their he she they them than then into about after before between during over under".split())
test_ids = {r["id"] for r in json.load(open(os.path.join(DS, "musique.json")))}
rows_ = [json.loads(l) for l in open(MT)]; rows_ = [r for r in rows_ if r["id"] not in test_ids]; random.shuffle(rows_)

def terms(text, limit=40):
    out = []
    for w in re.findall(r"[0-9A-Za-zÀ-ɏ一-鿿]+", text):
        lw = w.lower()
        if lw in STOP or len(lw) < 2 or lw in out: continue
        out.append(lw)
    return out[:limit]
def fts(text, limit=40):
    ts = terms(text, limit); return " OR ".join('"' + t.replace('"', '') + '"' for t in ts) if ts else '""'

db = sqlite3.connect(":memory:"); db.execute("CREATE VIRTUAL TABLE p USING fts5(title, body, tokenize='unicode61')")
key2id = {}; docs = []
for r in rows_:
    for pr in r["paragraphs"]:
        k = (pr["title"], pr["paragraph_text"])
        if k not in key2id: key2id[k] = len(docs); docs.append(k)
db.executemany("INSERT INTO p(rowid, title, body) VALUES (?,?,?)", [(i, t, b) for i, (t, b) in enumerate(docs)]); db.commit()
def search(text, k):
    try: return [row[0] for row in db.execute("SELECT rowid FROM p WHERE p MATCH ? ORDER BY bm25(p) LIMIT ?", (fts(text), k))]
    except sqlite3.OperationalError: return []

def state(q, ev, t, p):
    s = f"Original question: {q}\n"
    if ev:
        s += "Previously retrieved evidence (context only; may be wrong):\n"
        for i, (et, ep) in enumerate(ev[:EVM], 1): s += f"[{i}] {et}: {ep[:EVC]}\n"
    return s + f"Candidate title: {t}\nCandidate passage: {p}"

out = []; stats = {"q": 0, "r0": 0, "r1": 0, "r2": 0, "pos": 0}
for r in rows_[:N]:
    q = r["question"]; gold = [(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if p.get("is_supporting")]
    gold_ids = {key2id[g] for g in gold}
    top = search(q, 100); stats["q"] += 1
    def add(ev, cid, label, tag):
        t, p = docs[cid]; out.append({"state": state(q, ev, t, p), "questions": Q, "expected": {"verdict": bool(label)}}); stats[tag] += 1; stats["pos"] += int(bool(label))
    # round 0 (no evidence): all gold + 3 hard negs from top-20 + 2 from tail
    for g in gold_ids: add([], g, True, "r0")
    negs0 = [i for i in top[:20] if i not in gold_ids]; tail0 = [i for i in top[20:] if i not in gold_ids]
    for i in random.sample(negs0, min(3, len(negs0))) + random.sample(tail0, min(2, len(tail0))): add([], i, False, "r0")
    # evidence after round 0: gold found in top-100 + 2–5 wrong passages from top-20 (judge mistakes), shuffled
    found = [i for i in top if i in gold_ids]; wrong = random.sample(negs0, min(random.choice((2, 3, 4, 5)), len(negs0)))
    ev_ids = found + wrong; random.shuffle(ev_ids); ev = [docs[i] for i in ev_ids]
    # round 1: expansion from each evidence passage; candidates not already in evidence
    seen = set(ev_ids); r1 = []
    for i in ev_ids[:4]:
        r1 += [c for c in search(q + " " + docs[i][0] + " " + docs[i][1][:300], 40) if c not in seen]
    r1 = list(dict.fromkeys(r1))
    pos1 = [c for c in r1 if c in gold_ids]; neg1 = [c for c in r1 if c not in gold_ids]
    for c in pos1: add(ev, c, True, "r1")
    for c in gold_ids - set(found):  # gold the expansion should find: also as positive with evidence (teaches bridge→final)
        if c not in pos1: add(ev, c, True, "r1")
    for c in random.sample(neg1, min(5, len(neg1))): add(ev, c, False, "r1")
    # round 2: evidence grows with round-1 picks (gold found + 1–2 wrong), expansion again
    ev2_ids = ev_ids + pos1[:2] + random.sample(neg1, min(2, len(neg1))); random.shuffle(ev2_ids); ev2 = [docs[i] for i in ev2_ids]
    seen2 = set(ev2_ids); r2 = []
    for i in ev2_ids[-3:]:
        r2 += [c for c in search(q + " " + docs[i][0] + " " + docs[i][1][:300], 30) if c not in seen2]
    r2 = list(dict.fromkeys(r2)); pos2 = [c for c in r2 if c in gold_ids]; neg2 = [c for c in r2 if c not in gold_ids]
    for c in pos2: add(ev2, c, True, "r2")
    for c in random.sample(neg2, min(4, len(neg2))): add(ev2, c, False, "r2")
    # candidates that are themselves already in evidence (SAG2 re-scores them): keep their truth
    for c in random.sample(ev2_ids, min(2, len(ev2_ids))): add(ev2, c, c in gold_ids, "r2")
random.shuffle(out)
with open(OUT, "w") as f:
    for it in out: f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(json.dumps({"rows": len(out), **stats}), file=sys.stderr)
