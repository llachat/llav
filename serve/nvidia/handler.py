"""Hugging Face Inference Endpoints custom handler for llav (System One protocol inside the HF request envelope).

HF calls:  POST /  {"inputs": {"state": "...", "questions": {...}}, "parameters": {"noul": "rank"|"prob"}}
returns:   {"model": "llav-0.4", "answers": {qid: {"type":"noul","noul":..,"prob":..,"rank_score":..}}, "latency_ms": ..}

Single fused judge (this repo's weights) used for every round; evidence in `previously_retrieved_evidence` is rendered in the
v1 format the model was trained with. Runs on a 16 GB T4 in fp16 (≈8.5 GB weights) or any larger GPU in bf16.
"""
import json, math, os, time
CALIB = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib-llav-0.4.json"))) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib-llav-0.4.json")) else None
def _platt(p, ab):
    p = min(max(p, 1e-6), 1 - 1e-6); return 1 / (1 + math.exp(-(ab[0] * math.log(p / (1 - p)) + ab[1])))
def calibrate(probs, has_evidence):
    if not CALIB: return probs
    ab = CALIB.get("rounds" if has_evidence else "round0"); return [_platt(p, ab) for p in probs] if ab else probs
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
INSTRUCT = "Given a multi-hop question, judge whether the passage states a fact that is necessary to answer it (including a fact that identifies an intermediate entity in the answer chain)."
GENERAL_INSTRUCT = os.environ.get("LLAV_GENERAL_INSTRUCT", "Given a yes/no question about the Document, judge whether the correct answer is yes.")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def ranknorm(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i]); n = len(vals); r = [0.0] * n
    for k, i in enumerate(order): r[i] = k / max(n - 1, 1)
    return r


class EndpointHandler:
    def __init__(self, path: str = ""):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.dev == "cuda" else torch.float32
        if self.dev == "cuda" and torch.cuda.is_bf16_supported(): dtype = torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(path); self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(self.dev).eval()
        self.YES = self.tok.encode("yes", add_special_tokens=False)[0]; self.NO = self.tok.encode("no", add_special_tokens=False)[0]
        self.prefix = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
        self.batch = int(os.environ.get("LLAV_BATCH", "16")); self.max_len = int(os.environ.get("LLAV_MAX_LEN", "2048"))

    def _query(self, st: dict) -> str:
        q = str(st.get("original_question") or st.get("question") or "")
        sq = st.get("sub_questions") or []
        if sq: q += "\nSub-questions: " + " | ".join(f"{i}) {s}" for i, s in enumerate(sq, 1))
        ev = st.get("previously_retrieved_evidence") or []
        if ev:
            parts = []
            for e in ev[:10]:
                if isinstance(e, dict): parts.append(f"{e.get('title') or ''}: {str(e.get('passage') or e.get('content') or '')[:300]}")
                else: parts.append(str(e)[:300])
            q += "\nAlready retrieved evidence (context only): " + " || ".join(parts)
        return q

    def _enc(self, query: str, doc: str, instruct: str = INSTRUCT):
        ids = self.tok.encode(self.prefix + f"<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {doc}" + SUFFIX, add_special_tokens=False)
        if len(ids) >= self.max_len: ids = ids[: self.max_len - 40] + ids[-39:]
        return ids

    @torch.inference_mode()
    def _score(self, query: str, docs: list, items=None) -> list:
        """items: optional list of (query, doc, instruct) triples overriding (query, docs) — used for free-form questions."""
        seqs = [self._enc(*it) for it in items] if items is not None else [self._enc(query, d) for d in docs]; res = [0.0] * len(seqs)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        pad = self.tok.pad_token_id or 0
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]; chunk = [seqs[i] for i in idx]; L = max(len(x) for x in chunk)
            input_ids = torch.tensor([[pad] * (L - len(x)) + x for x in chunk], device=self.dev)
            attn = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in chunk], device=self.dev)
            logits = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            p = torch.softmax(torch.stack([logits[:, self.NO], logits[:, self.YES]], dim=1).float(), dim=1)[:, 1].tolist()
            for i, v in zip(idx, p): res[i] = v
        return res

    def __call__(self, data):
        t0 = time.time()
        body = data.get("inputs", data) if isinstance(data, dict) else {}
        if isinstance(body, str):
            try: body = json.loads(body)
            except Exception: body = {}
        params = (data.get("parameters") or {}) if isinstance(data, dict) else {}
        noul_mode = params.get("noul", os.environ.get("LLAV_NOUL", "rank"))
        state = body.get("state")
        if isinstance(state, str) and state.lstrip().startswith("{"):
            try: state = json.loads(state)
            except Exception: pass
        st = state if isinstance(state, dict) else {"original_question": str(state)}
        qs = body.get("questions") or {}; ids = list(qs)[:1000]; items = []; q0 = self._query(st)
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        for qid in ids:
            ins = (qs[qid] or {}).get("instructions") if isinstance(qs[qid], dict) else None
            if isinstance(ins, dict) and ("candidate_passage" in ins or "candidate_title" in ins):
                title = " / ".join(str(x) for x in (ins.get("candidate_title"), ins.get("candidate_heading")) if x)
                items.append((q0, f"{title}\n{ins.get('candidate_passage', '')}", INSTRUCT))
            else:  # free-form yes/no question: Query = the question, Document = the state
                qtxt = ins if isinstance(ins, str) else (ins.get("question") if isinstance(ins, dict) and ins.get("question") else json.dumps(ins, ensure_ascii=False))
                items.append((str(qtxt)[:2000], state_text[:6000], GENERAL_INSTRUCT))
        probs = self._score("", [], items) if items else []
        probs_raw = list(probs); probs = calibrate(probs, bool(st.get("previously_retrieved_evidence")))
        ranks = ranknorm(probs) if probs else []
        answers = {qid: {"type": "noul", "noul": float(ranks[i] if noul_mode == "rank" else probs[i]), "prob": float(probs[i]), "prob_raw": float(probs_raw[i]), "calibrated": CALIB is not None, "rank_score": float(ranks[i])} for i, qid in enumerate(ids)}
        return {"model": "llav-0.4", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}, "latency_ms": int((time.time() - t0) * 1000)}
