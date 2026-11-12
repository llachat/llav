# llav inference image (NVIDIA). Build: docker build -t llav-inference infra/inference
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip && pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
 && pip3 install "transformers>=4.51,<5" fastapi "uvicorn[standard]" hf_transfer huggingface_hub safetensors accelerate
WORKDIR /app
COPY server.py /app/server.py
# weights are pulled from the Hugging Face Hub at start (LLAV_ROUND0 / LLAV_ROUNDS = private repos; HF_TOKEN in env)
ENV PORT=8000 LLAV_BATCH=16 LLAV_MAX_LEN=2048
EXPOSE 8000
CMD ["python3", "/app/server.py"]
