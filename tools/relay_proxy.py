#!/usr/bin/env python3
"""Jev-compatible front for a local Laya server, so SAG-Wiki / SAG-Benchmark can use Laya as the decision model.

Why a proxy: Laya speaks the Jev wire protocol but (a) caps a request at 64 questions while SAG2 sends up to 100
candidates per round, and (b) budgets each *question* to `head_max_len` tokens (default 192) while SAG2 puts the whole
candidate passage inside the question, so everything gets truncated. The proxy splits/merges, raises the budgets, and
can re-shape the payload into Laya's native form (one state per candidate, batch endpoint) without touching either project.

Run:  ~/laya-venv/bin/python ~/models/laya-jev-proxy.py            (listens on 127.0.0.1:8001)
Env:  LAYA_URL=http://127.0.0.1:8000  LAYA_MODEL=typed-decisions  PROXY_PORT=8001  PROXY_CHUNK=64
      PROXY_MODE=passthrough|flat|flat2|batch|batchq  PROXY_MAXLEN=2048  PROXY_HEAD=1024  PROXY_LOG=~/models/evals/sag2/proxy-log.jsonl
Model spec from the client (SAG2's --jev-model / SAG-Wiki's Jev model field):  "<ckpt>@<mode>@<max_len>[:<head_max_len>]"
      e.g. "typed-decisions@batch@2048"  or  "multilingual@passthrough@4096:2048". Missing parts fall back to the env defaults.
Modes:
  passthrough  forward SAG2's payload as is (shared state, passage inside each question), chunked by 64
  flat / flat2 same, but nested `instructions` rendered as plain text (flat2 also shortens criteria)
  batch        Laya-native: one *state* per candidate (original question + evidence summary + candidate passage),
               one short shared noul question, sent to /v1/systemone/batch (64 states per call)
  batchq       like batch but the state carries no previously-retrieved evidence (question + candidate only)
"""
import asyncio, json, os, time
from typing import Any, Dict, List
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_MODEL = os.environ.get("LAYA_MODEL", "typed-decisions")
DEFAULT_MODE = os.environ.get("PROXY_MODE", "batch")
DEFAULT_MAXLEN = int(os.environ.get("PROXY_MAXLEN", "2048"))
DEFAULT_HEAD = int(os.environ.get("PROXY_HEAD", "1024"))
CHUNK = int(os.environ.get("PROXY_CHUNK", "64"))
EVIDENCE_CHARS = int(os.environ.get("PROXY_EVIDENCE_CHARS", "400"))
LOG = os.path.expanduser(os.environ.get("PROXY_LOG", "~/models/evals/sag2/proxy-log.jsonl"))
CHECKPOINTS = {"english", "multilingual", "typed-decisions", "vega"}
os.makedirs(os.path.dirname(LOG), exist_ok=True)
app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0), trust_env=False)

BATCH_QUESTION = ("Does the candidate passage itself state a fact that is necessary to answer the original question, "
                  "including a fact that identifies an intermediate entity in the answer chain? Use the previously retrieved "
                  "evidence only to connect facts. Topical association or background alone does not count. Use no outside knowledge.")
BATCH_CRITERIA = {"true": "The candidate passage supplies a necessary fact (or a necessary intermediate identification) for answering the original question.",
                  "false": "The candidate passage is only related background, concerns the wrong entity or relation, or the needed fact is only in other passages."}


def parse_model(name: str):
    parts = (name or "").split("@")
    ckpt = parts[0] if parts[0] else DEFAULT_MODEL  # any registered Laya checkpoint name (incl. LAYA_EXTRA_MODELS) passes through
    mode = parts[1] if len(parts) > 1 and parts[1] else DEFAULT_MODE
    maxlen, head = DEFAULT_MAXLEN, DEFAULT_HEAD
    if len(parts) > 2 and parts[2]:
        a, _, b = parts[2].partition(":")
        if a: maxlen = int(a)
        head = int(b) if b else min(head, maxlen // 2)
    return ckpt, mode, maxlen, min(head, maxlen - 64)


def flat_text(ins: Any) -> str:
    if not isinstance(ins, dict):
        return str(ins)
    parts = []
    if ins.get("question"): parts.append(str(ins["question"]))
    head = " / ".join(str(x) for x in (ins.get("candidate_title"), ins.get("candidate_heading")) if x)
    if head: parts.append(f"Candidate title: {head}")
    if ins.get("candidate_passage"): parts.append(f"Candidate passage: {ins['candidate_passage']}")
    for k, v in ins.items():
        if k not in ("question", "candidate_title", "candidate_heading", "candidate_passage"): parts.append(f"{k}: {v}")
    return "\n".join(parts)


def evidence_text(state: dict, limit_chars: int) -> str:
    ev = state.get("previously_retrieved_evidence") or []
    if not ev: return ""
    lines = ["Previously retrieved evidence (context only; may be wrong):"]
    for i, e in enumerate(ev[:10], 1):
        if isinstance(e, dict):
            t = e.get("title") or ""; p = str(e.get("passage") or e.get("content") or "")
            lines.append(f"[{i}] {t}: {p[:limit_chars]}")
        else:
            lines.append(f"[{i}] {str(e)[:limit_chars]}")
    return "\n".join(lines)


def render_questions(questions: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode == "passthrough":
        return questions
    out = {}
    for qid, q in questions.items():
        q2 = dict(q)
        ins = q.get("instructions") if isinstance(q.get("instructions"), dict) else {}
        if mode == "flat":
            q2["instructions"] = flat_text(q.get("instructions"))
        elif mode == "flat2":
            head = " / ".join(str(x) for x in (ins.get("candidate_title"), ins.get("candidate_heading")) if x)
            q2["instructions"] = (BATCH_QUESTION + "\n" + (f"Candidate title: {head}\n" if head else "") + f"Candidate passage: {ins.get('candidate_passage', '')}")
            q2["criteria"] = BATCH_CRITERIA
        out[qid] = q2
    return out


async def post(path: str, payload: dict) -> dict:
    r = await client.post(f"{LAYA_URL}{path}", json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"laya {r.status_code}: {r.text[:300]}")
    return r.json()


def merge_usage(usage: dict, u: dict):
    usage["input_tokens"] += int(u.get("input_tokens") or 0); usage["output_tokens"] += int(u.get("output_tokens") or 0)
    usage["truncated_questions"] += list(u.get("truncated_questions") or []); usage["truncated"] = usage["truncated"] or bool(u.get("truncated"))
    usage["state_tokens_dropped"] += int(u.get("state_tokens_dropped") or 0)


async def run_shared(ckpt, mode, maxlen, head, state, questions):
    rendered = render_questions(questions, mode); ids = list(rendered)
    chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)] or [[]]
    payloads = [{"model": ckpt, "state": state, "questions": {k: rendered[k] for k in ch}, "max_len": maxlen, "head_max_len": head} for ch in chunks]
    results = await asyncio.gather(*(post("/v1/systemone", p) for p in payloads))
    answers, usage = {}, {"input_tokens": 0, "output_tokens": 0, "truncated_questions": [], "truncated": False, "state_tokens_dropped": 0}
    for res in results:
        for k, v in (res.get("answers") or {}).items():
            answers[k] = {"type": "noul", "noul": v.get("noul"), "confidence": v.get("confidence")} if isinstance(v, dict) else v
        merge_usage(usage, res.get("usage") or {})
    return answers, usage


async def run_batch(ckpt, mode, maxlen, head, state, questions):
    st = state if isinstance(state, dict) else {"original_question": str(state)[:2000]}
    oq = st.get("original_question") or st.get("question") or ""
    ev = "" if mode == "batchq" else evidence_text(st, EVIDENCE_CHARS)
    ids = list(questions); states: List[str] = []
    subq = st.get("sub_questions") or []
    subt = ("Sub-questions:\n" + "\n".join(f"{i}) {q}" for i, q in enumerate(subq, 1)) + "\n") if subq else ""
    for qid in ids:
        ins = questions[qid].get("instructions") if isinstance(questions[qid].get("instructions"), dict) else {}
        headt = " / ".join(str(x) for x in (ins.get("candidate_title"), ins.get("candidate_heading")) if x)
        txt = f"Original question: {oq}\n" + subt + (ev + "\n" if ev else "") + f"Candidate title: {headt}\nCandidate passage: {ins.get('candidate_passage', '')}"
        states.append(txt)
    q = {"verdict": {"type": "noul", "instructions": BATCH_QUESTION, "criteria": BATCH_CRITERIA}}
    chunks = [list(range(i, min(i + CHUNK, len(ids)))) for i in range(0, len(ids), CHUNK)]
    payloads = [{"model": ckpt, "states": [states[i] for i in ch], "questions": q, "max_len": maxlen, "head_max_len": head} for ch in chunks]
    results = await asyncio.gather(*(post("/v1/systemone/batch", p) for p in payloads))
    answers, usage = {}, {"input_tokens": 0, "output_tokens": 0, "truncated_questions": [], "truncated": False, "state_tokens_dropped": 0}
    for ch, res in zip(chunks, results):
        items = res.get("results") or []
        for i, item in zip(ch, items):
            a = (item.get("answers") or {}).get("verdict") or {}
            answers[ids[i]] = {"type": "noul", "noul": a.get("noul"), "confidence": a.get("confidence")}
            u = item.get("usage") or {}
            if u.get("state_tokens_dropped"): usage["truncated_questions"].append(ids[i]); usage["truncated"] = True
            merge_usage(usage, {k: u.get(k) for k in ("input_tokens", "output_tokens", "state_tokens_dropped")})
    return answers, usage


@app.get("/health")
async def health():
    try:
        r = await client.get(f"{LAYA_URL}/health"); up = r.json()
    except Exception as e:
        up = {"error": str(e)}
    return {"proxy": "ok", "laya": up, "defaults": {"model": DEFAULT_MODEL, "mode": DEFAULT_MODE, "max_len": DEFAULT_MAXLEN, "head_max_len": DEFAULT_HEAD}, "chunk": CHUNK}


OPENROUTER_URL = os.environ.get("OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions")
OPENROUTER_KEY_FILE = os.path.expanduser(os.environ.get("OPENROUTER_KEY_FILE", "~/.config/openrouter.env"))
relay = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=20.0), trust_env=True)


def openrouter_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY", "")
    if not k and os.path.exists(OPENROUTER_KEY_FILE):
        for line in open(OPENROUTER_KEY_FILE):
            if line.startswith("OPENROUTER_API_KEY="): k = line.split("=", 1)[1].strip()
    return k


async def run_relay(body: dict):
    """Relay the untouched payload to the real Jev (OpenRouter) so its per-candidate scores land in the same log."""
    r = await relay.post(OPENROUTER_URL, json=body, headers={"Authorization": "Bearer " + openrouter_key(), "HTTP-Referer": "http://localhost", "X-Title": "sag2-local-eval"})
    if r.status_code != 200:
        raise RuntimeError(f"openrouter {r.status_code}: {r.text[:300]}")
    d = r.json(); u = d.get("usage") or {}
    return d, {"input_tokens": int(u.get("input_tokens") or 0), "output_tokens": int(u.get("output_tokens") or 0), "truncated_questions": [], "truncated": False, "state_tokens_dropped": 0, "cost": u.get("cost")}


@app.post("/v1/systemone")
@app.post("/systemone")
@app.post("/decisions")
@app.post("/v1/decisions")
async def systemone(request: Request):
    body = await request.json(); t0 = time.time()
    questions = body.get("questions") or {}; state = body.get("state")
    if isinstance(state, str) and state.lstrip().startswith("{"):
        try: state = json.loads(state)
        except Exception: pass
    if str(body.get("model", "")).startswith(("typesafe/", "google/", "openai/")):   # relay closed decision models through OpenRouter
        try:
            d, usage = await run_relay(body)
        except Exception as e:
            return JSONResponse({"error": {"message": str(e)}}, status_code=502)
        answers = d.get("answers") or {}; model_tag = str(d.get("model") or body.get("model"))
        try:
            st = state if isinstance(state, dict) else {}
            with open(LOG, "a") as f:
                f.write(json.dumps({"ts": time.time(), "model": model_tag, "n": len(questions), "latency_s": round(time.time() - t0, 3), "truncated": False, "n_truncated": 0,
                                    "state_tokens_dropped": 0, "cost": usage.get("cost"), "question": str(st.get("original_question", ""))[:300],
                                    "prev_evidence": len(st.get("previously_retrieved_evidence") or []),
                                    "nouls": {k: v.get("noul") for k, v in answers.items() if isinstance(v, dict)}}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return JSONResponse(d)
    ckpt, mode, maxlen, head = parse_model(body.get("model", ""))
    if mode == "raw":
        # forward the SAG2 payload unchanged (model name = ckpt), only chunked by 64 questions: for Jev-protocol servers (Kev, Strands...)
        try:
            ids = list(questions); chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)] or [[]]
            results = await asyncio.gather(*(post("/v1/systemone", {"model": ckpt, "state": body.get("state"), "questions": {k: questions[k] for k in ch}}) for ch in chunks))
            answers = {}; usage = {"input_tokens": 0, "output_tokens": 0, "truncated_questions": [], "truncated": False, "state_tokens_dropped": 0}
            for res in results:
                for k, v in (res.get("answers") or {}).items(): answers[k] = v
                merge_usage(usage, res.get("usage") or {})
        except Exception as e:
            return JSONResponse({"error": {"message": str(e)}}, status_code=502)
        model_tag = f"{ckpt}@raw"; out = {"model": model_tag, "answers": answers, "usage": usage}
        try:
            st = state if isinstance(state, dict) else {}
            with open(LOG, "a") as f:
                f.write(json.dumps({"ts": time.time(), "model": model_tag, "n": len(questions), "latency_s": round(time.time() - t0, 3), "truncated": False, "n_truncated": 0, "state_tokens_dropped": 0,
                                    "question": str(st.get("original_question", ""))[:300], "prev_evidence": len(st.get("previously_retrieved_evidence") or []),
                                    "nouls": {k: v.get("noul") for k, v in answers.items() if isinstance(v, dict)}}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return JSONResponse(out)
    try:
        if mode in ("batch", "batchq"):
            answers, usage = await run_batch(ckpt, mode, maxlen, head, state, questions)
        else:
            answers, usage = await run_shared(ckpt, mode, maxlen, head, body.get("state"), questions)
    except Exception as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=502)
    model_tag = f"laya/{ckpt}@{mode}@{maxlen}:{head}"
    out = {"model": model_tag, "answers": answers, "usage": usage}
    try:
        st = state if isinstance(state, dict) else {}
        with open(LOG, "a") as f:
            f.write(json.dumps({"ts": time.time(), "model": model_tag, "n": len(questions), "latency_s": round(time.time() - t0, 3),
                                "truncated": usage["truncated"], "n_truncated": len(usage["truncated_questions"]), "state_tokens_dropped": usage["state_tokens_dropped"],
                                "question": str(st.get("original_question", ""))[:300], "prev_evidence": len(st.get("previously_retrieved_evidence") or []),
                                "nouls": {k: v.get("noul") for k, v in answers.items() if isinstance(v, dict)}}, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return JSONResponse(out)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PROXY_PORT", "8001")), log_level="warning")
