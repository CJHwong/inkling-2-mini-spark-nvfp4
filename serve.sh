#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
preset=fast
if [[ ${1:-} == q8 || ${1:-} == fast ]]; then
    preset=$1
    shift
fi
vision=false
if [[ $# == 1 && $1 == --vision ]]; then
    vision=true
elif [[ $# != 0 ]]; then
    echo 'Usage: ./serve.sh [q8|fast] [--vision]' >&2
    exit 2
fi
if [[ ! -x .venv/bin/vllm || ! -f model/DOWNLOAD_VERIFIED.json ]]; then
    echo 'Run ./setup.sh first.' >&2
    exit 1
fi

export CUDA_HOME=/usr/local/cuda
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface"
export VLLM_CACHE_ROOT="$PWD/.cache/$preset/vllm"
export TRITON_CACHE_DIR="$PWD/.cache/$preset/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/$preset/torchinductor"
export CUDA_CACHE_PATH="$PWD/.cache/$preset/cuda"
export FLASHINFER_WORKSPACE_BASE="$PWD/.cache/$preset"
mkdir -p "$CUDA_CACHE_PATH"
if [[ -d .runtime/headers/usr/include ]]; then
    export C_INCLUDE_PATH="$PWD/.runtime/headers/usr/include/python3.12:$PWD/.runtime/headers/usr/include"
fi

unset PYTHONPATH INKLING2_TARGET_FP4 INKLING2_HEAD_FP4 INKLING2_EXPERT_FP4
export INKLING2_DRAFT_QUANT=int8 INKLING2_DRAFT_QUANT_LAYERS=1
cache=auto
package=.venv/lib/python3.12/site-packages/vllm
if [[ $preset == fast ]]; then
    if [[ ! -f .runtime/fast/vllm/__init__.py ]]; then
        echo 'Run ./setup.sh runtime to install the fast preset.' >&2
        exit 1
    fi
    export PYTHONPATH="$PWD/.runtime/fast"
    package=.runtime/fast/vllm
    export INKLING2_TARGET_FP4=1 INKLING2_HEAD_FP4=1 INKLING2_EXPERT_FP4=1
    export INKLING2_DRAFT_QUANT=nvfp4-native
    cache=fp8
fi

batch_tokens=4096
media_limit='{"image":0,"video":0}'
if [[ $vision == true ]]; then
    if ! grep -Fq 'import vit_torch_sdpa_wrapper' "$package/models/inkling2/nvidia/vision.py"; then
        echo 'Run ./setup.sh runtime to install the vision patch.' >&2
        exit 1
    fi
    batch_tokens=5120
    media_limit='{"image":8,"video":0}'
fi

exec .venv/bin/vllm serve "$PWD/model" \
    --kv-cache-dtype "$cache" --served-model-name inkling --tensor-parallel-size 1 --dtype bfloat16 \
    --max-model-len 262144 --gpu-memory-utilization 0.61 --max-num-seqs 8 \
    --max-num-batched-tokens "$batch_tokens" --limit-mm-per-prompt "$media_limit" \
    --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
    --safetensors-load-strategy eager --moe-backend humming \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
    --host 127.0.0.1 --port 8020
