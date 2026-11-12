"""llav inference on Modal (serverless NVIDIA, scales to zero = free-tier abuse costs nothing when idle).
deploy:   modal deploy infra/inference/modal_app.py
secrets:  modal secret create llav-inference LLAV_INTERNAL_SECRET=<same as the Worker's INFERENCE_SECRET>
weights:  modal volume put llav-weights ~/models/rerank-ft/fused/llav-0.4-round0 /round0 ; … llav-0.4-rounds /rounds
The deployed web endpoint URL (printed by `modal deploy`) goes into the Worker as INFERENCE_URL.
"""
import os
import modal

# weights live in the Modal volume "llav-weights" (uploaded from the Mac with `modal volume put`), mounted at /weights
ROUND0 = os.environ.get("LLAV_ROUND0", "/weights/round0")
ROUNDS = os.environ.get("LLAV_ROUNDS", "/weights/rounds")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers>=4.51,<5", "fastapi", "uvicorn", "hf_transfer", "huggingface_hub", "safetensors", "accelerate")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "LLAV_ROUND0": ROUND0, "LLAV_ROUNDS": ROUNDS})
    .add_local_file(os.path.join(os.path.dirname(__file__), "server.py"), "/app/server.py")
)
app = modal.App("llav-inference")
weights = modal.Volume.from_name("llav-weights", create_if_missing=True)
hf_cache = modal.Volume.from_name("llav-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",                    # 24 GB; L4 never scheduled for this workspace (probes: A10G 220 s, T4 306 s, L4 none)
    secrets=[modal.Secret.from_name("llav-inference")],
    volumes={"/weights": weights, "/root/.cache/huggingface": hf_cache},
    scaledown_window=1200,         # stay warm 20 min after the last request (scheduling takes minutes here), then scale to zero
    timeout=600,
    max_containers=2,              # hard cap on spend; raise when real traffic justifies it
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/app")
    import server   # loads both models at container start
    return server.app
