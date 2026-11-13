#!/usr/bin/env python3
"""llav v0.3 server (per-round routing)

  --round0  "soupC:3,qwen4b:1,prism08:1,qwen06:1"   members+weights used when the request carries NO previously_retrieved_evidence
  --rounds  "v4:1"                                     members+weights used when evidence is present (rounds 1–2); default = round0 set
  per-member evidence format: "|style=v1" or "|style=v4" (must match how that adapter was trained)

llav v0.2 server: Jev/System One-compatible `/v1/systemone` backed by an MLX **ensemble with a cascade**.

  members  : several MLX cross-encoders (Qwen3-Reranker 0.6B/4B, Prism, fused/LoRA variants), each with a weight;
             final score = weighted average of per-request rank-normalised scores (what beat Jev on the held-out 150)
  cascade  : an optional cheap prefilter model scores all candidates first; only the top --keep go to the heavy members,
             the rest keep the prefilter's (rank-normalised, scaled below the kept block) score. Cuts latency ~3×.

Member spec: "name=/path/to/model[|adapter=/path][|w=3][|system=...]"   (semicolon-separated list in --members)
run: ~/rerank-venv/bin/python ~/llav/llav-serve2.py --members "ft4b=/…/fused|w=3;qwen4b=/…/qwen3-reranker-4b;prism=/…/prism-0.8b|system=Judge whether the Document meets the requirements based on the Query and the Instruct provided. ;qwen06=/…/qwen3-reranker-0.6b" \
       --prefilter qwen06 --keep 30 --port 8020
"""
import argparse, json, os, time, hmac
import mlx.core as mx
import mlx_lm.utils as _U; _U.MODEL_REMAPPING["qwen3_5_text"] = "qwen3_5"
from mlx_lm.utils import load
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--members", required=True); ap.add_argument("--prefilter", default=""); ap.add_argument("--keep", type=int, default=30)
ap.add_argument("--port", type=int, default=8020); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--noul", default="rank", choices=["rank","prob"], help="what the noul field carries: rank = request-relative rank-normalised ensemble score (what SAG2 was validated with); prob = weighted mean of member probabilities"); ap.add_argument("--round0", default=""); ap.add_argument("--rounds", default="")
ap.add_argument("--general-instruct", default="Given a yes/no question about the Document, judge whether the correct answer is yes.", help="instruct used when a question is free-form text (not a retrieval candidate): Query = the question, Document = the state"); ap.add_argument("--prob-member", default="", help="member whose probability feeds the prob field (e.g. soupC); default = weighted mean of member probabilities"); ap.add_argument("--calib", default="", help="JSON with Platt params {\"round0\":[a,b],\"rounds\":[a,b]}: prob -> sigmoid(a*logit(prob)+b); raw kept in prob_raw"); ap.add_argument("--name", default="llav-0.3"); ap.add_argument("--evidence-chars", type=int, default=300); ap.add_argument("--evidence-style", default="v1", choices=["v1","v4"]); ap.add_argument("--log", default=os.path.expanduser("~/llav/logs/llav-serve2-requests.jsonl"))
ap.add_argument("--instruct", default="Given a multi-hop question, judge whether the passage states a fact that is necessary to answer it (including a fact that identifies an intermediate entity in the answer chain).")
a = ap.parse_args()
GENERAL = a.general_instruct
import math as _m
CALIB = json.load(open(os.path.expanduser(a.calib))) if a.calib else None
def _platt(p, ab):
    p = min(max(p, 1e-6), 1 - 1e-6); z = ab[0] * _m.log(p / (1 - p)) + ab[1]; return 1 / (1 + _m.exp(-z))
def calibrate(probs, has_evidence):
    if not CALIB: return probs
    ab = CALIB.get("rounds" if has_evidence else "round0") or CALIB.get("platt")
    return [_platt(p, ab) for p in probs] if ab else probs

try: mx.set_cache_limit(2 << 30)
except Exception: pass
DEFAULT_SYSTEM = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Member:
    def __init__(self, spec):
        name, _, rest = spec.partition("="); parts = rest.split("|"); self.name = name.strip(); self.path = parts[0].strip()
        self.adapter = ""; self.w = 1.0; self.system = DEFAULT_SYSTEM; self.style = a.evidence_style
        for p in parts[1:]:
            k, _, v = p.partition("=")
            if k == "adapter": self.adapter = v
            elif k == "w": self.w = float(v)
            elif k == "system": self.system = v
            elif k == "style": self.style = v
        self.model, self.tok = load(self.path, adapter_path=self.adapter or None)
        tm = self.model.language_model if hasattr(self.model, "language_model") else self.model
        self.inner = tm.model
        self.head = (lambda h: self.inner.embed_tokens.as_linear(h)) if getattr(tm.args, "tie_word_embeddings", False) else tm.lm_head
        self.YES = self.tok.encode("yes", add_special_tokens=False)[0]; self.NO = self.tok.encode("no", add_special_tokens=False)[0]
        self.prefix = "<|im_start|>system\n" + self.system + "<|im_end|>\n<|im_start|>user\n"
        print(f"loaded member {self.name} w={self.w} style={self.style} ({self.path}{' + ' + self.adapter if self.adapter else ''})", flush=True)

    def enc(self, query, doc, instruct=None):
        ids = self.tok.encode(self.prefix + f"<Instruct>: {instruct or a.instruct}\n<Query>: {query}\n<Document>: {doc}" + SUFFIX, add_special_tokens=False)
        if len(ids) >= a.max_len: ids = ids[: a.max_len - 40] + ids[-39:]
        return ids

    def score(self, pairs):
        out = []; seqs = [self.enc(*pr) for pr in pairs]
        for i in range(0, len(seqs), a.batch):
            chunk = seqs[i:i + a.batch]; L = max(len(s) for s in chunk)
            arr = mx.array([s + [0] * (L - len(s)) for s in chunk]); last = mx.array([len(s) - 1 for s in chunk])
            h = self.inner(arr)[mx.arange(len(chunk)), last]; lg = self.head(h)
            p = mx.softmax(mx.stack([lg[:, self.NO], lg[:, self.YES]], axis=1).astype(mx.float32), axis=1)[:, 1]; mx.eval(p); out += p.tolist()
            try: mx.clear_cache()
            except Exception: pass
        return out


members = [Member(s) for s in a.members.split(";") if s.strip()]
byname = {m.name: m for m in members}
def parse_set(spec):
    out = []
    for item in spec.split(","):
        if not item.strip(): continue
        n, _, w = item.partition(":"); m = byname[n.strip()]; out.append((m, float(w) if w else m.w))
    return out
SET0 = parse_set(a.round0) if a.round0 else [(m, m.w) for m in members]
SET12 = parse_set(a.rounds) if a.rounds else SET0
pre = next((m for m in members if m.name == a.prefilter), None) if a.prefilter else None
heavy = [m for m in members if m is not pre] if pre else members


def ranknorm(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i]); n = len(vals); r = [0.0] * n
    for k, i in enumerate(order): r[i] = k / max(n - 1, 1)
    return r


def build_query(st, style):
    q = str(st.get("original_question") or st.get("question") or "")
    sq = st.get("sub_questions") or []
    if sq: q += "\nSub-questions: " + " | ".join(f"{i}) {s}" for i, s in enumerate(sq, 1))
    ev = st.get("previously_retrieved_evidence") or []
    if ev:
        parts = []
        for i, e in enumerate(ev[:(8 if style == "v4" else 10)], 1):
            tag = f"[{i}] " if style == "v4" else ""
            if isinstance(e, dict): parts.append(f"{tag}{e.get('title') or ''}: {str(e.get('passage') or e.get('content') or '')[:a.evidence_chars]}")
            else: parts.append(f"{tag}{str(e)[:a.evidence_chars]}")
        q += "\nAlready retrieved evidence (context only): " + " || ".join(parts)
    return q


app = FastAPI()


@app.get("/health")
async def health(): return {"status": "ok", "model": a.name, "round0": [(m.name, w) for m, w in SET0], "rounds": [(m.name, w) for m, w in SET12], "prefilter": a.prefilter, "keep": a.keep}


@app.get("/v1/models")
async def models(): return {"data": [{"id": a.name, "object": "model"}]}


INTERNAL_SECRET = os.environ.get("LLAV_INTERNAL_SECRET", "")
@app.post("/v1/systemone")
@app.post("/systemone")
@app.post("/decisions")
async def systemone(request: Request):
    if INTERNAL_SECRET and not hmac.compare_digest(request.headers.get("X-Llav-Internal", ""), INTERNAL_SECRET):
        from fastapi.responses import JSONResponse as _JR; return _JR({"error": "unauthorized"}, status_code=401)
    body = await request.json(); t0 = time.time()
    state = body.get("state")
    if isinstance(state, str) and state.lstrip().startswith("{"):
        try: state = json.loads(state)
        except Exception: pass
    st = state if isinstance(state, dict) else {"original_question": str(state)}
    qs = body.get("questions") or {}; ids = list(qs); has_ev = bool(st.get("previously_retrieved_evidence"))
    active = SET12 if has_ev else SET0; docs = []
    for qid in ids:
        ins = qs[qid].get("instructions")
        if isinstance(ins, dict) and ("candidate_passage" in ins or "candidate_title" in ins):
            title = " / ".join(str(x) for x in (ins.get("candidate_title"), ins.get("candidate_heading")) if x)
            docs.append(f"{title}\n{ins.get('candidate_passage', '')}")
        else:
            docs.append(str(state)[:6000])
    # free-form questions (no candidate passage): the question text becomes the Query, the state the Document, under the general instruct
    gen_q = [None] * len(ids)
    for j, qid in enumerate(ids):
        ins = qs[qid].get("instructions")
        if not (isinstance(ins, dict) and ("candidate_passage" in ins or "candidate_title" in ins)):
            txt = ins if isinstance(ins, str) else (ins.get("question") if isinstance(ins, dict) and ins.get("question") else json.dumps(ins, ensure_ascii=False))
            gen_q[j] = str(txt)[:2000]
    qcache = {}
    def pairs_for(m):
        if m.style not in qcache: qcache[m.style] = build_query(st, m.style)
        return [((gen_q[j], d, GENERAL) if gen_q[j] is not None else (qcache[m.style], d)) for j, d in enumerate(docs)]
    pairs = docs
    n = len(pairs); final = [0.0] * n; detail = {}
    if n:
        keep_idx = list(range(n)); pre_rank = None
        if pre is not None and n > a.keep:
            ps = pre.score(pairs_for(pre)); pre_rank = ranknorm(ps); detail[pre.name] = ps
            keep_idx = sorted(range(n), key=lambda i: -ps[i])[: a.keep]
        acc = [0.0] * len(keep_idx); accp = [0.0] * len(keep_idx); wsum = 0.0; pre_w = None
        for m, w in active:
            if m is pre and pre_rank is not None: pre_w = w; continue
            pm = pairs_for(m); sub = [pm[i] for i in keep_idx]
            sc = m.score(sub); detail[m.name] = dict(zip(keep_idx, sc)); rn = ranknorm(sc)
            for j in range(len(sub)): acc[j] += w * rn[j]; accp[j] += w * sc[j]
            wsum += w
        if pre_w is not None:
            # the prefilter also votes inside the kept block
            prs = ranknorm([pre_rank[i] for i in keep_idx]); pps = detail[pre.name]
            for j, i in enumerate(keep_idx): acc[j] += pre_w * prs[j]; accp[j] += pre_w * pps[i]
            wsum += pre_w
        kept = {i: acc[j] / (wsum or 1.0) for j, i in enumerate(keep_idx)}; keptp = {i: accp[j] / (wsum or 1.0) for j, i in enumerate(keep_idx)}
        probs = [keptp.get(i, (detail[pre.name][i] if pre is not None and pre.name in detail else 0.0)) for i in range(n)]
        if pre_rank is not None and len(keep_idx) < n:
            lo = min(kept.values()) if kept else 0.0
            for i in range(n):
                final[i] = kept[i] if i in kept else lo * pre_rank[i]  # non-kept stay below every kept candidate, ordered by the prefilter
        else:
            for i in range(n): final[i] = kept.get(i, 0.0)
    if not n: probs = []
    if a.prob_member and a.prob_member in detail and isinstance(detail[a.prob_member], dict):
        pm = detail[a.prob_member]; probs = [float(pm.get(i, probs[i])) for i in range(len(probs))]
    probs_raw = list(probs); probs = calibrate(probs, bool(st.get("previously_retrieved_evidence")))
    answers = {qid: {"type": "noul", "noul": float(min(max(final[i] if a.noul == "rank" else probs[i], 0.0), 1.0)), "rank_score": float(min(max(final[i], 0.0), 1.0)), "prob_raw": float(min(max(probs_raw[i], 0.0), 1.0)), "calibrated": CALIB is not None, "prob": float(min(max(probs[i], 0.0), 1.0))} for i, qid in enumerate(ids)}
    out = {"model": a.name, "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}, "latency_ms": int((time.time() - t0) * 1000)}
    try:
        with open(a.log, "a") as f:
            f.write(json.dumps({"ts": time.time(), "n": n, "set": "rounds" if has_ev else "round0", "kept": len(keep_idx) if n else 0, "latency_s": round(time.time() - t0, 3), "question": str(st.get("original_question", ""))[:200],
                                "prev_evidence": len(st.get("previously_retrieved_evidence") or []), "nouls": {k: v["noul"] for k, v in answers.items()}}, ensure_ascii=False) + "\n")
    except Exception: pass
    return JSONResponse(out)


uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
