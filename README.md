# llav — open decision models

**A probability for every question.** llav is an open-weights *decision model*: send a state (your context) and a dictionary of typed questions, get a calibrated probability per question in one round trip. It speaks the System One protocol (`POST /v1/systemone`), so code written for Jev works by changing the base URL.

- Website & free API: https://llav.app  ·  Weights: https://huggingface.co/llavapp  ·  License: Apache-2.0
- v0.4 = Qwen3-Reranker-4B base + our LoRA (listwise objective, adapter soups), specialised for **evidence judging in multi-hop retrieval** (SAG2-compatible).

## Benchmarks (measured by us, same candidates, same pipeline, same day — `docs/benchmarks.json`)

Judge-level, 100 BM25 candidates per question, MuSiQue (AUC / R@10 / AllGold@10):

| split | llav v0.4 | Jev 1.14 | Gemini Decision 1 |
|---|---|---|---|
| held-out q50–200 | 0.970 / 0.919 / 0.864 | 0.969 / 0.924 / 0.864 | 0.966 / 0.879 / 0.779 |
| final test q200–350 | 0.984 / 0.945 / 0.872 (ensemble 0.984 / 0.949 / 0.893) | 0.983 / 0.951 / 0.899 | 0.981 / 0.933 / 0.859 |

End-to-end SAG2 Recall@10 (50 questions each): MuSiQue llav 0.812 · Jev 0.933 · Gemini 0.895; 2Wiki 0.975 / 0.975 / 0.975; HotpotQA 0.99 / 1.00 / 0.99. llav's MuSiQue gap is loop dynamics (evidence lock-in), not judge quality — see `docs/research-plan-v5-zh.md`; a set-level final-round judge is in training. Where Jev still leads: evidence-round AUC (0.987 vs 0.976) and probability calibration (Brier 0.013 vs 0.046).

## Layout

| dir | what |
|---|---|
| `serve/mlx/` | Apple-silicon server (MLX): per-round member routing, cascade prefilter, anti-lock-in option (`--exclude-self`), `--noul rank|prob` |
| `serve/nvidia/` | CUDA server (transformers) + Dockerfile + Modal app + Hugging Face Inference Endpoints `handler.py` |
| `train/` | listwise LoRA trainer, set-level judge trainer, data builders (round-aware rows, real SAG2 rounds, set data), adapter soups |
| `eval/` | judge benchmark, MLX reranker benchmark, per-round replay, set-level benchmark, ensembles, calibration, selective-prediction, hop analysis |
| `tools/` | System One relay proxy (Jev/Gemini via OpenRouter, Laya, Kev, Strands) used for apples-to-apples comparisons |
| `docs/` | benchmark data, research plan (zh), data-flywheel architecture (zh) |

Paths inside the scripts default to the authors' machine layout (`~/models/...`); every script takes explicit `--model-path/--details/--dataset-dir` style flags — pass your own.

## Run the model

**Apple silicon (MLX):**
```bash
pip install mlx-lm fastapi uvicorn
python serve/mlx/llav_serve.py --members "llav=/path/to/fused-llav-0.4-round0|w=1|style=v1" --port 8020 --noul prob
curl localhost:8020/v1/systemone -H 'Content-Type: application/json' -d '{"state":"{\"original_question\":\"Who directed Forrest Gump?\"}","questions":{"c1":{"type":"noul","instructions":{"candidate_title":"Robert Zemeckis","candidate_passage":"Robert Zemeckis directed Forrest Gump."}}}}'
```
**NVIDIA (one 24 GB GPU, or 16 GB with a single model):**
```bash
LLAV_ROUND0=llavapp/llav-0.4-round0 LLAV_ROUNDS=llavapp/llav-0.4-rounds LLAV_INTERNAL_SECRET=change-me python serve/nvidia/server.py
```
**Hugging Face Inference Endpoints:** the model repos contain `handler.py`; create an endpoint with task *Custom*.

## Train / evaluate
See the docstrings at the top of each script. Typical loop: `train/build-*.py` → `train/train-rerank-listwise.py` or `train/train-set-judge.py` → `eval/mlx-rerank-bench.py` (judge level) → `eval/round-bench.py` (evidence rounds) → SAG2 end-to-end.

## Status
Preview. Weights are released with the v0.4 public launch (Apache-2.0). Roadmap: set-level judging, calibration layer, 9B/27B backbones, image and audio inputs, fine-tuning on customer outcomes via the API feedback loop.
