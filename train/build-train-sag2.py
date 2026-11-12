#!/usr/bin/env python3
"""Build a *training-split* SAG2 dataset dir (same format as the benchmark's dataset/) so the SAG2 pipeline + Jev can be run on
training questions: the resulting traces give, per round, the exact candidates, the evidence passed, Jev's scores and gold labels
= round-aware distillation data for llav v5. Questions are disjoint from the 1,000-question eval sets (by id and by text).
Corpus per dataset = union of the paragraphs of N_CORPUS sampled training questions (~10k docs, like the benchmark corpora);
only the first N_Q of them are written as questions (the rest just make retrieval realistically hard).
usage: build-train-sag2.py OUT_DIR [--n-corpus 1000] [--n-q 300] [--seed 7]"""
import json, random, sys, os
import pyarrow.parquet as pq
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
args = dict(zip(sys.argv[2::2], sys.argv[3::2])); N_CORPUS = int(args.get("--n-corpus", 1000)); N_Q = int(args.get("--n-q", 300)); SEED = int(args.get("--seed", 7))
RAW = os.path.expanduser("~/models/evals/sag2/data/raw"); DATA = os.path.expanduser("~/models/evals/sag2/data")
def eval_sets(n):
    e = json.load(open(f"{DATA}/eval-ids-{n}.json")); return set(e["ids"]), set(q.strip() for q in e["questions"])

def finish(name, samples, docs_of):
    ids, qs = eval_sets(name)
    samples = [s for s in samples if s.get("_id", s.get("id")) not in ids and s["question"].strip() not in qs]
    random.Random(SEED).shuffle(samples); samples = samples[:N_CORPUS]
    corpus = {}
    for s in samples:
        for title, text in docs_of(s): corpus.setdefault((title.strip(), text), None)
    questions = samples[:N_Q]
    # every gold doc must be in the corpus (same strings the loader builds)
    json.dump(questions, open(f"{out}/{name}.json", "w"), ensure_ascii=False)
    json.dump([{"title": t, "text": x} for t, x in corpus], open(f"{out}/{name}_corpus.json", "w"), ensure_ascii=False)
    print(f"{name}: {len(questions)} questions, corpus {len(corpus)} docs (from {len(samples)} sampled train questions)")

# MuSiQue (answerable train split)
mus = [json.loads(l) for l in open(f"{DATA}/musique_ans_v1.0_train.jsonl")]
finish("musique", mus, lambda s: [(p["title"], p["paragraph_text"]) for p in s["paragraphs"]])

# HotpotQA distractor train (struct columns -> benchmark's list format)
hot = []
for f in sorted(x for x in os.listdir(RAW) if x.startswith("hotpotqa_")):
    for r in pq.read_table(f"{RAW}/{f}").to_pylist():
        hot.append({"_id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"], "level": r["level"],
                    "supporting_facts": [[t, i] for t, i in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])],
                    "context": [[t, s] for t, s in zip(r["context"]["title"], r["context"]["sentences"])]})
finish("hotpotqa", hot, lambda s: [(t, "".join(sents)) for t, sents in s["context"]])

# 2Wiki train (JSON-string columns)
wiki = []
for r in pq.read_table(f"{RAW}/xanhho_2WikiMultihopQA__train.parquet").to_pylist():
    wiki.append({"_id": r["_id"], "type": r["type"], "question": r["question"], "context": json.loads(r["context"]),
                 "supporting_facts": json.loads(r["supporting_facts"]), "evidences": json.loads(r["evidences"]), "answer": r["answer"]})
finish("2wikimultihopqa", wiki, lambda s: [(t, " ".join(sents)) for t, sents in s["context"]])
open(f"{out}/README.md", "w").write("Training-split SAG2 datasets built by build-train-sag2.py (disjoint from the eval sets). Used only to collect Jev traces for llav v5 training.\n")
