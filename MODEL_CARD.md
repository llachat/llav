# llav-0.4 model card
- **Base:** Qwen/Qwen3-Reranker-4B (Apache-2.0). **Method:** LoRA (rank 8, 16 layers) with pointwise yes/no CE + listwise CE over each question's candidate group; checkpoint averaging ("soups").
- **Variants:** `llav-0.4-round0` (no-evidence rounds) and `llav-0.4-rounds` (evidence-conditioned rounds, v4 format). Fused full weights, bf16.
- **Training data:** MuSiQue train split (gold evidence labels), BM25 hard negatives, simulated/real SAG2 rounds (HotpotQA, 2Wiki train splits). No eval-set questions.
- **Eval:** MuSiQue/2Wiki/HotpotQA eval subsets, SAG2 pipeline; see README and `docs/benchmarks.json`.
- **Intended use:** judging whether a passage is necessary evidence for a question (RAG/multi-hop retrieval), relevance-style noul questions. **Not** a general typed-decision model yet; English-centric; probabilities are ranking-reliable but not calibrated (use `noul` for ordering, calibrate `prob` on your data).
- **License:** Apache-2.0.
