#!/bin/bash
# Hunyuan3D texture painting with GPU management
# Stops ComfyUI container to free VRAM, runs texture painting, restarts ComfyUI
set -e

INPUT_IMAGE="${1:-/mnt/agents/output/huaqing_front_crop.png}"
OUTPUT_GLB="${2:-/mnt/agents/output/tex_pbr_output.glb}"
GEO_GLB="${3:-}"  # Optional: existing geometry GLB. If empty, shape gen runs first.
RENDER_SIZE="${4:-512}"
TEXTURE_SIZE="${5:-512}"

COMFYUI_CONTAINER="comfyui-primary"
COMFYUI_WAS_RUNNING=false

echo "[gpu-mgr] Input: $INPUT_IMAGE"
echo "[gpu-mgr] Output: $OUTPUT_GLB"
echo "[gpu-mgr] Render: ${RENDER_SIZE}x${TEXTURE_SIZE}"

# Step 0: Stop ComfyUI container to free VRAM
if docker ps --format '{{.Names}}' | grep -q "$COMFYUI_CONTAINER"; then
    echo "[gpu-mgr] Stopping $COMFYUI_CONTAINER to free VRAM..."
    docker stop "$COMFYUI_CONTAINER" 2>/dev/null
    COMFYUI_WAS_RUNNING=true
    # Wait for GPU memory to be released
    for i in $(seq 1 15); do
        sleep 2
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
        echo "[gpu-mgr]   GPU1: ${GPU_MEM}MiB"
        if [ "$GPU_MEM" -lt 500 ]; then
            break
        fi
    done
    echo "[gpu-mgr] ComfyUI stopped, GPU freed"
fi

# Step 1: Shape generation (if no geometry provided)
if [ -z "$GEO_GLB" ]; then
    GEO_GLB="/mnt/agents/output/_tmp_shape_$$.glb"
    echo "[gpu-mgr] Running shape generation..."
    docker exec kais-gold-team bash -c "cd /app && python3 scripts/hunyuan3d_infer.py \
        --input '$INPUT_IMAGE' \
        --model mini \
        --texture-mode none \
        --device cuda:0 \
        --steps 50 \
        --model-dir /data/models/Hunyuan3D-2mini \
        --output '$GEO_GLB' 2>&1" || {
        echo "[gpu-mgr] ERROR: Shape generation failed"
        if [ "$COMFYUI_WAS_RUNNING" = true ]; then
            docker start "$COMFYUI_CONTAINER" 2>/dev/null
        fi
        exit 1
    }
    echo "[gpu-mgr] Shape generation done: $GEO_GLB"
fi

# Step 2: Texture painting (standalone script)
echo "[gpu-mgr] Running texture painting..."
docker exec kais-gold-team bash -c "cd /app && python3 scripts/hunyuan3d_texture_only.py \
    --geo '$GEO_GLB' \
    --image '$INPUT_IMAGE' \
    --output '$OUTPUT_GLB' \
    --render-size $RENDER_SIZE \
    --texture-size $TEXTURE_SIZE 2>&1"

TEX_RC=$?
if [ $TEX_RC -ne 0 ]; then
    echo "[gpu-mgr] ERROR: Texture painting failed (rc=$TEX_RC)"
else
    echo "[gpu-mgr] Texture painting done: $OUTPUT_GLB"
fi

# Step 3: Cleanup temp geometry
if [ -f "/mnt/agents/output/_tmp_shape_$$.glb" ]; then
    rm -f "/mnt/agents/output/_tmp_shape_$$.glb"
fi

# Step 4: Restart ComfyUI container
if [ "$COMFYUI_WAS_RUNNING" = true ]; then
    echo "[gpu-mgr] Restarting $COMFYUI_CONTAINER..."
    docker start "$COMFYUI_CONTAINER" 2>/dev/null
    echo "[gpu-mgr] ComfyUI restarting (takes ~30s to load models)"
fi

# Check result
if [ -f "$OUTPUT_GLB" ]; then
    SIZE=$(stat -c%s "$OUTPUT_GLB")
    echo "[gpu-mgr] SUCCESS: ${SIZE} bytes"
    exit 0
else
    echo "[gpu-mgr] FAILED: no output file"
    exit 1
fi
