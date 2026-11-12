#!/usr/bin/env python3
"""Build Laya fine-tuning data for the SAG2 evidence judgment, from HotpotQA / 2Wiki (train) and MuSiQue (eval only).

Each row: {"state": <text exactly as laya-jev-proxy batch mode renders it>, "questions": {"verdict": noul question},
           "expected": {"verdict": true|false}}
Positives = a question's supporting paragraphs; negatives = the distractor paragraphs shipped with the same question
(HotpotQA distractor setting: 10 paragraphs per question, 2 gold). A share of rows also carries 1–2 other gold
paragraphs as "previously retrieved evidence" so the model learns the SAG2 round-2 situation (context connects facts,
candidate is still judged on its own content).

usage: build-train.py OUT_DIR [--eval-only] [--musique-train PATH.jsonl] [--max-musique N] [--neg-per-q K]
With --musique-train, OUT_DIR/train-v2.jsonl = hotpot + 2wiki + up to N MuSiQue *training-split* questions (ids disjoint from the test set).
writes OUT_DIR/train.jsonl (hotpotqa + 2wiki), OUT_DIR/train-hotpot-only.jsonl, OUT_DIR/eval-musique.jsonl, OUT_DIR/eval-2wiki.jsonl
"""
import json, os, random, sys
DS = os.environ.get("SAG2_DATASET_DIR", "/Users/williamsing/models/evals/sag2/bench/SAG-Benchmark/sag2/dataset")
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
random.seed(20261111)
# keep in sync with laya-jev-proxy.py
BATCH_QUESTION = ("Does the candidate passage itself state a fact that is necessary to answer the original question, "
                  "including a fact that identifies an intermediate entity in the answer chain? Use the previously retrieved "
                  "evidence only to connect facts. Topical association or background alone does not count. Use no outside knowledge.")
BATCH_CRITERIA = {"true": "The candidate passage supplies a necessary fact (or a necessary intermediate identification) for answering the original question.",
                  "false": "The candidate passage is only related background, concerns the wrong entity or relation, or the needed fact is only in other passages."}
Q = {"verdict": {"type": "noul", "instructions": BATCH_QUESTION, "criteria": BATCH_CRITERIA}}
EV_CHARS = 400


def state_text(question, title, passage, evidence, subq=None):
    s = f"Original question: {question}\n"
    if subq:
        s += "Sub-questions:\n" + "".join(f"{i}) {q}\n" for i, q in enumerate(subq, 1))
    if evidence:
        s += "Previously retrieved evidence (context only; may be wrong):\n"
        for i, (t, p) in enumerate(evidence[:10], 1): s += f"[{i}] {t}: {p[:EV_CHARS]}\n"
    return s + f"Candidate title: {title}\nCandidate passage: {passage}"


def hotpot_like(path):
    """HotpotQA / 2Wiki distractor format: context = [[title, [sentences]]], supporting_facts = [[title, sent_id]]."""
    for r in json.load(open(path)):
        gold_titles = {t for t, _ in r["supporting_facts"]}
        paras = [(t, " ".join(s).strip()) for t, s in r["context"]]
        pos = [(t, p) for t, p in paras if t in gold_titles]; neg = [(t, p) for t, p in paras if t not in gold_titles]
        yield r["question"], pos, neg


def musique(path, limit=0):
    rows_ = [json.loads(l) for l in open(path)] if path.endswith(".jsonl") else json.load(open(path))
    if limit: rows_ = rows_[:limit]
    for r in rows_:
        pos = [(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if p.get("is_supporting")]
        neg = [(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if not p.get("is_supporting")]
        yield r["question"], pos, neg


def rows(gen, neg_per_q=6, evidence_share=0.4):
    out = []
    for item in gen:
        q, pos, neg = item[0], item[1], item[2]; subq = item[3] if len(item) > 3 else None
        if not pos: continue
        negs = random.sample(neg, min(neg_per_q, len(neg)))
        for t, p in pos:
            ev = [x for x in pos if x != (t, p)]
            use_ev = ev if (ev and random.random() < evidence_share) else []
            out.append({"state": state_text(q, t, p, use_ev, subq), "questions": Q, "expected": {"verdict": True}})
        for t, p in negs:
            use_ev = random.sample(pos, min(len(pos), random.choice((1, 2)))) if random.random() < evidence_share else []
            out.append({"state": state_text(q, t, p, use_ev, subq), "questions": Q, "expected": {"verdict": False}})
    random.shuffle(out); return out


def dump(name, items):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    npos = sum(1 for it in items if it["expected"]["verdict"]); print(f"{name}: {len(items)} rows ({npos} true / {len(items)-npos} false)")


NEG = int(sys.argv[sys.argv.index("--neg-per-q") + 1]) if "--neg-per-q" in sys.argv else 6
hot = rows(hotpot_like(os.path.join(DS, "hotpotqa.json")), neg_per_q=NEG)
wiki = rows(hotpot_like(os.path.join(DS, "2wikimultihopqa.json")), neg_per_q=NEG)
mus = rows(musique(os.path.join(DS, "musique.json")), neg_per_q=4, evidence_share=0.4)
if "--eval-only" not in sys.argv:
    dump("train.jsonl", hot + wiki); dump("train-hotpot-only.jsonl", hot)
dump("eval-musique.jsonl", mus)
dump("eval-2wiki.jsonl", wiki[:1500])
if "--musique-train" in sys.argv:
    mt_path = sys.argv[sys.argv.index("--musique-train") + 1]
    mx = int(sys.argv[sys.argv.index("--max-musique") + 1]) if "--max-musique" in sys.argv else 0
    test_ids = {r["id"] for r in json.load(open(os.path.join(DS, "musique.json")))}
    mt_rows = [json.loads(l) for l in open(mt_path) if json.loads(l)["id"] not in test_ids]
    random.shuffle(mt_rows)
    if mx: mt_rows = mt_rows[:mx]
    DEC = "--decomp" in sys.argv
    def gen():
        for r in mt_rows:
            pos = [(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if p.get("is_supporting")]
            neg = [(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if not p.get("is_supporting")]
            subq = [d["question"] for d in r.get("question_decomposition", [])] if DEC else None
            yield r["question"], pos, neg, subq
    mt = rows(gen(), neg_per_q=NEG)
    print(f"musique-train questions used: {len(mt_rows)} decomp={DEC}")
    suffix = "-decomp" if DEC else ""
    dump(f"train-v2{suffix}.jsonl", hot + wiki + mt)
    dump(f"train-musique-only{suffix}.jsonl", mt)
