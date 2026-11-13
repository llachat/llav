#!/usr/bin/env python
"""Calibration layer for llav: fit temperature (and Platt a,b) on one dump, evaluate Brier / log-loss / ECE on another.
Dumps are per-question rows {scores:{id: s}, gold:[ids]}; scores may be logits (ctx30 protocol) or probabilities. Only
candidates with a real model score are used (ctx30 fills the tail with sentinel values below the minimum: we drop them via --topk).
  python fit-calib.py --fit dumps-rounds/setctx-0001500-50.jsonl --eval dumps-rounds/p9v5-01000-150.jsonl --logits --topk 30
"""
import argparse, json, math
ap = argparse.ArgumentParser(); ap.add_argument("--fit", required=True); ap.add_argument("--eval", required=True)
ap.add_argument("--logits", action="store_true", help="scores are logits (yes-no); otherwise probabilities"); ap.add_argument("--topk", type=int, default=0)
ap.add_argument("--bins", type=int, default=10); ap.add_argument("--out", default="")
a = ap.parse_args()
def sig(x): return 1 / (1 + math.exp(-x))
def logit(p): p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))
def load(f):
    X, Y = [], []
    for l in open(f):
        e = json.loads(l); g = set(int(x) for x in e["gold"]); sc = {int(k): float(v) for k, v in e["scores"].items()}
        ids = sorted(sc, key=lambda k: -sc[k]); ids = ids[: a.topk] if a.topk else ids
        for k in ids: X.append(sc[k] if a.logits else logit(sc[k])); Y.append(1.0 if k in g else 0.0)
    return X, Y
def metrics(P, Y):
    n = len(P); brier = sum((p - y) ** 2 for p, y in zip(P, Y)) / n; ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)) for p, y in zip(P, Y)) / n
    ece = 0.0
    for b in range(a.bins):
        lo, hi = b / a.bins, (b + 1) / a.bins; idx = [i for i, p in enumerate(P) if lo <= p < hi or (b == a.bins - 1 and p == 1.0)]
        if idx: ece += len(idx) / n * abs(sum(Y[i] for i in idx) / len(idx) - sum(P[i] for i in idx) / len(idx))
    return {"n": n, "pos_rate": round(sum(Y) / n, 4), "brier": round(brier, 4), "logloss": round(ll, 4), "ece": round(ece, 4)}
def fit(X, Y, platt):
    # minimise log-loss of sig(a*x+b) by Newton steps (temperature: b fixed 0)
    A, B = 1.0, 0.0
    for _ in range(200):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for x, y in zip(X, Y):
            p = sig(A * x + B); d = p - y; w = p * (1 - p)
            g_a += d * x; g_b += d; h_aa += w * x * x; h_ab += w * x; h_bb += w
        if platt:
            det = h_aa * h_bb - h_ab * h_ab + 1e-12; dA = (h_bb * g_a - h_ab * g_b) / det; dB = (h_aa * g_b - h_ab * g_a) / det
        else: dA = g_a / (h_aa + 1e-12); dB = 0.0
        A -= dA; B -= dB
        if abs(dA) + abs(dB) < 1e-7: break
    return A, B
Xf, Yf = load(a.fit); Xe, Ye = load(a.eval)
raw = metrics([sig(x) for x in Xe], Ye)
T, _ = fit(Xf, Yf, False); Ap, Bp = fit(Xf, Yf, True)
temp = metrics([sig(T * x) for x in Xe], Ye); platt = metrics([sig(Ap * x + Bp) for x in Xe], Ye)
res = {"fit_rows": len(Xf), "eval_rows": len(Xe), "raw": raw, "temperature": {"T": round(T, 4), **temp}, "platt": {"a": round(Ap, 4), "b": round(Bp, 4), **platt}}
print(json.dumps(res, indent=1))
if a.out: json.dump({"temperature": T, "platt": [Ap, Bp]}, open(a.out, "w"))
