#!/usr/bin/env python3
"""llav inference server for NVIDIA GPUs (cloud). System One protocol, same semantics as the Mac MLX backend:
  round 0 (no previously_retrieved_evidence) -> model ROUND0  (fused soupC: Qwen3-Reranker-4B + v1 LoRA, "v1" query format)
  evidence rounds                            -> model ROUNDS  (fused v4 soup A, "v4" query format)
Scores = P(yes) from the yes/no logits at the last position (Qwen3-Reranker head). Batched, bf16, torch SDPA.
Env: LLAV_ROUND0 (HF repo or dir), LLAV_ROUNDS, LLAV_INTERNAL_SECRET (required header X-Llav-Internal), LLAV_BATCH (16),
     LLAV_MAX_LEN (2048), LLAV_NOUL (rank|prob). Health: GET /health. Port 8000.
"""
import json, os, time, hmac, asyncio, threading
import math as _m
CALIB = json.load(open(os.environ["LLAV_CALIB"])) if os.environ.get("LLAV_CALIB") and os.path.exists(os.environ["LLAV_CALIB"]) else None
def _platt(p, ab):
    p = min(max(p, 1e-6), 1 - 1e-6); z = ab[0] * _m.log(p / (1 - p)) + ab[1]; return 1 / (1 + _m.exp(-z))
def calibrate(probs, has_evidence):
    if not CALIB: return probs
    ab = CALIB.get("rounds" if has_evidence else "round0") or CALIB.get("platt")
    return [_platt(p, ab) for p in probs] if ab else probs
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM

SECRET = os.environ.get("LLAV_INTERNAL_SECRET", "")
BATCH = int(os.environ.get("LLAV_BATCH", "16")); MAX_LEN = int(os.environ.get("LLAV_MAX_LEN", "2048")); NOUL = os.environ.get("LLAV_NOUL", "rank")
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEV == "cuda" else torch.float16 if DEV == "mps" else torch.float32
SYSTEM = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
INSTRUCT = "Given a multi-hop question, judge whether the passage states a fact that is necessary to answer it (including a fact that identifies an intermediate entity in the answer chain)."
GENERAL_INSTRUCT = os.environ.get("LLAV_GENERAL_INSTRUCT", "Given a yes/no question about the Document, judge whether the correct answer is yes.")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Judge:
    def __init__(self, path: str, style: str):
        self.style = style
        self.tok = AutoTokenizer.from_pretrained(path); self.tok.padding_side = "left"
        q = os.environ.get("LLAV_QUANT", "")   # "8bit" / "4bit": serve an 8B judge on a 12 GB card
        if q:
            from transformers import BitsAndBytesConfig
            qc = BitsAndBytesConfig(load_in_8bit=True) if q == "8bit" else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE, quantization_config=qc, device_map={"": 0}).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE).to(DEV).eval()
        if os.environ.get("LLAV_ADAPTER"):   # LoRA on top of a (quantised) base, no merge needed
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, os.environ["LLAV_ADAPTER"]).eval()
        self.YES = self.tok.encode("yes", add_special_tokens=False)[0]; self.NO = self.tok.encode("no", add_special_tokens=False)[0]
        self.prefix = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
        print(f"loaded {path} style={style} on {DEV} {DTYPE}", flush=True)

    def enc(self, query: str, doc: str, instruct: str = INSTRUCT):
        ids = self.tok.encode(self.prefix + f"<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {doc}" + SUFFIX, add_special_tokens=False)
        if len(ids) >= MAX_LEN: ids = ids[: MAX_LEN - 40] + ids[-39:]
        return ids

    @torch.inference_mode()
    def score(self, query: str, docs: list[str], items=None) -> list[float]:
        """items: optional (query, doc, instruct) triples overriding (query, docs) — free-form questions use the general instruct."""
        seqs = [self.enc(*it) for it in items] if items is not None else [self.enc(query, d) for d in docs]; out: list[float] = []
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))   # length-sorted batches: less padding
        res = [0.0] * len(seqs)
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]; chunk = [seqs[i] for i in idx]; L = max(len(x) for x in chunk)
            input_ids = torch.tensor([[self.tok.pad_token_id or 0] * (L - len(x)) + x for x in chunk], device=DEV)
            attn = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in chunk], device=DEV)
            logits = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            two = torch.stack([logits[:, self.NO], logits[:, self.YES]], dim=1).float()
            p = torch.softmax(two, dim=1)[:, 1].tolist()
            for i, v in zip(idx, p): res[i] = v
        if DEV == "cuda": torch.cuda.empty_cache()   # give memory back between requests (12 GB cards, Windows pages on overflow)
        return res


def build_query(st: dict, style: str) -> str:
    q = str(st.get("original_question") or st.get("question") or "")
    sq = st.get("sub_questions") or []
    if sq: q += "\nSub-questions: " + " | ".join(f"{i}) {s}" for i, s in enumerate(sq, 1))
    ev = st.get("previously_retrieved_evidence") or []
    if ev:
        parts = []
        for i, e in enumerate(ev[:(8 if style == "v4" else 10)], 1):
            tag = f"[{i}] " if style == "v4" else ""
            if isinstance(e, dict): parts.append(f"{tag}{e.get('title') or ''}: {str(e.get('passage') or e.get('content') or '')[:300]}")
            else: parts.append(f"{tag}{str(e)[:300]}")
        q += "\nAlready retrieved evidence (context only): " + " || ".join(parts)
    return q


def ranknorm(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i]); n = len(vals); r = [0.0] * n
    for k, i in enumerate(order): r[i] = k / max(n - 1, 1)
    return r


round0 = Judge(os.environ.get("LLAV_ROUND0", "llav/llav-0.4-round0"), "v1")
rounds = Judge(os.environ.get("LLAV_ROUNDS", "llav/llav-0.4-rounds"), "v4") if os.environ.get("LLAV_ROUNDS") else round0
app = FastAPI()


def authed(request: Request) -> bool:
    return bool(SECRET) and hmac.compare_digest(request.headers.get("X-Llav-Internal", ""), SECRET)


GPU_LOCK = threading.Lock()   # one forward pass at a time on the card; the event loop stays free for health checks and queued requests
def _score_locked(judge, items):
    with GPU_LOCK:
        return judge.score("", [], items)

@app.get("/health")
async def health(request: Request):
    if not authed(request): return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {"status": "ok", "model": "llav-0.4", "device": DEV, "round0": os.environ.get("LLAV_ROUND0"), "rounds": os.environ.get("LLAV_ROUNDS")}


@app.post("/v1/systemone")
async def systemone(request: Request):
    if not authed(request): return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json(); t0 = time.time()
    state = body.get("state")
    if isinstance(state, str) and state.lstrip().startswith("{"):
        try: state = json.loads(state)
        except Exception: pass
    st = state if isinstance(state, dict) else {"original_question": str(state)}
    qs = body.get("questions") or {}; ids = list(qs)[:1000]
    judge = rounds if st.get("previously_retrieved_evidence") else round0
    q0 = build_query(st, judge.style); state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False); items = []
    for qid in ids:
        ins = (qs[qid] or {}).get("instructions")
        if isinstance(ins, dict) and ("candidate_passage" in ins or "candidate_title" in ins):
            title = " / ".join(str(x) for x in (ins.get("candidate_title"), ins.get("candidate_heading")) if x)
            items.append((q0, f"{title}\n{ins.get('candidate_passage', '')}", INSTRUCT))
        else:  # free-form yes/no question: Query = the question, Document = the state
            qtxt = ins if isinstance(ins, str) else (ins.get("question") if isinstance(ins, dict) and ins.get("question") else json.dumps(ins, ensure_ascii=False))
            items.append((str(qtxt)[:2000], state_text[:6000], GENERAL_INSTRUCT))
    probs = (await asyncio.to_thread(_score_locked, judge, items)) if items else []
    probs_raw = list(probs); probs = calibrate(probs, bool(st.get("previously_retrieved_evidence")))
    ranks = ranknorm(probs) if probs else []
    answers = {qid: {"type": "noul", "noul": float(ranks[i] if NOUL == "rank" else probs[i]), "prob": float(probs[i]), "prob_raw": float(probs_raw[i]), "calibrated": CALIB is not None, "rank_score": float(ranks[i])} for i, qid in enumerate(ids)}
    return {"model": "llav-0.4", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}, "latency_ms": int((time.time() - t0) * 1000)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="warning")
