#!/usr/bin/env python3
"""Selective prediction ("automation rate at a precision target"): for a judge's scores, what share of candidate decisions
can be auto-accepted (score ≥ t) / auto-rejected (score ≤ t') while keeping the accepted set at ≥P precision and the rejected
set at ≥R negative-precision; the rest go to a fallback (LLM or human). Also the full threshold table.
usage: selective-eval.py --member TAG=dump.jsonl ... [--precision 0.95 0.99] [--neg-precision 0.995] [--platt-on dump.jsonl]"""
import argparse, json, math
ap = argparse.ArgumentParser(); ap.add_argument("--member", action="append", required=True); ap.add_argument("--precision", nargs="*", type=float, default=[0.90, 0.95, 0.99])
ap.add_argument("--neg-precision", type=float, default=0.995); ap.add_argument("--platt-on", default=""); a = ap.parse_args()
def load(f):
    P, Y = [], []
    for l in open(f):
        e = json.loads(l); g = set(e["gold"])
        for k, v in e["scores"].items(): P.append(min(max(float(v), 1e-6), 1 - 1e-6)); Y.append(1 if int(k) in g else 0)
    return P, Y
def platt(Pfit, Yfit):
    # 1-D logistic regression on logit(p): p' = sigmoid(a*logit(p)+b), fitted by a few Newton steps
    X = [math.log(p / (1 - p)) for p in Pfit]; A, B = 1.0, 0.0
    for _ in range(50):
        g = [1 / (1 + math.exp(-(A * x + B))) for x in X]
        ga = sum((gi - y) * x for gi, y, x in zip(g, Yfit, X)); gb = sum(gi - y for gi, y in zip(g, Yfit))
        haa = sum(gi * (1 - gi) * x * x for gi, x in zip(g, X)) + 1e-6; hbb = sum(gi * (1 - gi) for gi in g) + 1e-6; hab = sum(gi * (1 - gi) * x for gi, x in zip(g, X))
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12: break
        dA = (hbb * ga - hab * gb) / det; dB = (haa * gb - hab * ga) / det; A -= dA; B -= dB
        if abs(dA) + abs(dB) < 1e-6: break
    return lambda p: 1 / (1 + math.exp(-(A * math.log(p / (1 - p)) + B)))
for m in a.member:
    tag, _, f = m.partition("="); P, Y = load(f)
    if a.platt_on:
        cal = platt(*load(a.platt_on)); P = [min(max(cal(p), 1e-6), 1 - 1e-6) for p in P]
    n = len(P); pos = sum(Y); pairs = sorted(zip(P, Y), key=lambda t: -t[0])
    out = [f"{tag:14s} n={n} positives={pos} ({pos/n:.1%})"]
    # auto-accept: largest prefix (by score desc) with precision >= target
    for tgt in a.precision:
        tp = 0; best = None
        for i, (p, y) in enumerate(pairs, 1):
            tp += y
            if tp / i >= tgt: best = (i, p, tp)
        if best: out.append(f"  accept@P>={tgt:.2f}: threshold>={best[1]:.3f} covers {best[0]/n:.1%} of decisions, recall of positives {best[2]/pos:.1%}")
        else: out.append(f"  accept@P>={tgt:.2f}: unreachable")
    # auto-reject: largest suffix (score asc) with negative precision >= target
    tn = 0; best = None
    for i, (p, y) in enumerate(reversed(pairs), 1):
        tn += 1 - y
        if tn / i >= a.neg_precision: best = (i, p, tn)
    if best: out.append(f"  reject@NP>={a.neg_precision:.3f}: threshold<={best[1]:.3f} covers {best[0]/n:.1%} of decisions (misses {pos - (pos - (best[0]-best[2]))} positives → {(best[0]-best[2])/pos:.1%} of positives wrongly rejected)")
    print("\n".join(out))
