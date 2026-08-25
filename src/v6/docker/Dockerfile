FROM python:3.11-slim

WORKDIR /app

# Use TUNA mirror for apt
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git ffmpeg libsndfile1 gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple || \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install CosyVoice + Chatterbox + GPT-SoVITS dependencies
RUN pip install --no-cache-dir \
    conformer==0.3.2 \
    diffusers \
    hydra-core==1.3.2 \
    HyperPyYAML==1.2.3 \
    "librosa==0.10.2" \
    modelscope \
    omegaconf==2.3.0 \
    soundfile \
    "transformers==4.51.3" \
    onnxruntime-gpu \
    openai-whisper \
    inflect \
    wget \
    pyarrow \
    wetext \
    pyworld \
    matcha-tts \
    tqdm \
    torchcodec \
    x-transformers \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# Chatterbox-Turbo — install --no-deps to avoid numpy/librosa conflict
# (chatterbox requires numpy<1.26 but we use 1.26+; other deps already installed)
RUN pip install --no-cache-dir --no-deps chatterbox-tts && \
    pip install --no-cache-dir --no-deps \
    safetensors \
    sentencepiece \
    encodec \
    # GPT-SoVITS dependencies
    gradio \
    cn2an \
    pypinyin \
    jieba_fast \
    wordsegment \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# Patch librosa numba caching issue
RUN find /usr/local/lib/python3.11/site-packages/librosa -name "*.py" -exec \
    sed -i 's/@jit(nopython=True, cache=True)/@jit(nopython=True, cache=False)/g; s/@jit(cache=True)/@jit(cache=False)/g; s/cache=True, /cache=False, /g' {} +

# Clone CosyVoice repo (model downloaded separately via volume or init script)
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice || \
    echo "CosyVoice clone failed - mount or download manually"

# Install ACE-Step (music generation) — COPY from local clone to avoid build-time network issues
# nano-vllm is a bundled local package required by ace-step (not on PyPI)
# Both installed with --no-deps to avoid GitHub-hosted flash-attn wheel download in build.
# v1.5: ACE-Step source install removed. Music generation now runs via
# ComfyUI workflows submitted by Node-layer routes (src/routes/v1/ace/).
# gold-team no longer bundles acestep Python sources, /opt/acestep dir,
# or any ACESTEP_* env vars.

# ── Hunyuan3D (image-to-3D) ────────────────────────────────────────────────
# pip package for shape generation + PBR texture painting
RUN pip install --no-cache-dir hy3dgen trimesh \
    -i https://pypi.tuna.tsinghua.edu.cn/simple || \
    pip install --no-cache-dir hy3dgen trimesh

# ── Kimodo (motion diffusion) ──────────────────────────────────────────────
# Install with --no-deps + SKIP_MOTION_CORRECTION to avoid C++ build + dep conflicts
# (kimodo requires transformers==5.1 which conflicts with CosyVoice's 4.51.3)
ENV SKIP_MOTION_CORRECTION_IN_SETUP=1
RUN pip install --no-cache-dir --no-deps "kimodo[all] @ git+https://github.com/nv-tlabs/kimodo.git" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple || \
    pip install --no-cache-dir --no-deps "kimodo[all] @ git+https://github.com/nv-tlabs/kimodo.git"

# Patch: use NousResearch/Meta-Llama-3-8B-Instruct (open mirror) instead of gated meta-llama
COPY scripts/patch_kimodo_llama_mirror.py /tmp/patch_kimodo.py
RUN python3 /tmp/patch_kimodo.py && rm /tmp/patch_kimodo.py

# Copy v6 module
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN groupadd -r kais && useradd -r -g kais kais && \
    chown -R kais:kais /app /opt/CosyVoice
USER kais

ENV PORT=8002
ENV COSYVOICE_ROOT=/opt/CosyVoice
ENV COSYVOICE_MODEL_DIR=/opt/CosyVoice/pretrained_models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
ENV COSYVOICE_LLM_WEIGHT=llm.rl.pt
ENV MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
  CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "src.v6.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
