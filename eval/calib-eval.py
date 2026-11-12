#!/usr/bin/env python3
"""Calibration of judge scores as probabilities (what an API customer sees): Brier, log-loss, ECE (15 bins), and the same after
temperature scaling fitted on another dump (or on itself, reported as 'oracle'). Inputs are the per-question dumps ({scores:{id:p}, gold}).
usage: calib-eval.py --member TAG=dump.jsonl ... [--fit-on dump.jsonl] [--bins 15]"""
import argparse, json, math
ap = argparse.ArgumentParser(); ap.add_argument("--member", action="append", required=True); ap.add_argument("--fit-on", default=""); ap.add_argument("--bins", type=int, default=15); a = ap.parse_args()
def load(f):
    P, Y = [], []
    for l in open(f):
        e = json.loads(l); g = set(e["gold"])
        for k, v in e["scores"].items(): P.append(min(max(float(v), 1e-6), 1 - 1e-6)); Y.append(1 if int(k) in g else 0)
    return P, Y
def logit(p): return math.log(p / (1 - p))
def sig(x): return 1 / (1 + math.exp(-x))
def metrics(P, Y, bins):
    n = len(P); brier = sum((p - y) ** 2 for p, y in zip(P, Y)) / n; ll = -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(P, Y)) / n
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins; idx = [i for i, p in enumerate(P) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx: ece += len(idx) / n * abs(sum(P[i] for i in idx) / len(idx) - sum(Y[i] for i in idx) / len(idx))
    return brier, ll, ece
def fit_temp(P, Y):
    best, bt = 1e9, 1.0
    for t in [x / 20 for x in range(2, 200)]:
        Q = [min(max(sig(logit(p) / t), 1e-6), 1 - 1e-6) for p in P]; ll = -sum(y * math.log(q) + (1 - y) * math.log(1 - q) for q, y in zip(Q, Y)) / len(Q)
        if ll < best: best, bt = ll, t
    return bt
for m in a.member:
    tag, _, f = m.partition("="); P, Y = load(f); pos = sum(Y) / len(Y)
    b, ll, e = metrics(P, Y, a.bins)
    T = fit_temp(*load(a.fit_on)) if a.fit_on else fit_temp(P, Y)
    Q = [min(max(sig(logit(p) / T), 1e-6), 1 - 1e-6) for p in P]; b2, ll2, e2 = metrics(Q, Y, a.bins)
    print(f"{tag:28s} n={len(P):6d} pos={pos:.3f} meanP={sum(P)/len(P):.3f} | raw: Brier={b:.4f} logloss={ll:.4f} ECE={e:.4f} | temp T={T:.2f}{' (fit on '+a.fit_on.split('/')[-1]+')' if a.fit_on else ' (self)'}: Brier={b2:.4f} logloss={ll2:.4f} ECE={e2:.4f}")
