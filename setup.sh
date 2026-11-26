#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

case ${1:-all} in
    all|runtime|model) scope=${1:-all} ;;
    *) echo 'Usage: ./setup.sh [runtime|model]' >&2; exit 2 ;;
esac
if [[ $(uname -m) != aarch64 || $(uname -s) != Linux ]]; then
    echo 'Use Linux ARM64 on the DGX Spark.' >&2
    exit 1
fi

recipe_url=https://github.com/sxuff/inkling-2-mini-spark.git
recipe_commit=35a4fd8344f01148d35c899cc107aab0c537ce3d
recipe="$PWD/.runtime/recipe"
mkdir -p .runtime
if [[ ! -d "$recipe/.git" ]]; then
    git init -q "$recipe"
fi
if [[ $(git -C "$recipe" rev-parse --verify HEAD 2>/dev/null || true) != "$recipe_commit" ]]; then
    git -C "$recipe" fetch --depth 1 "$recipe_url" "$recipe_commit"
    git -C "$recipe" checkout --detach FETCH_HEAD
fi
if [[ $(git -C "$recipe" rev-parse HEAD) != "$recipe_commit" ]]; then
    echo 'The recipe commit does not match the pin.' >&2
    exit 1
fi

if [[ $scope != runtime ]]; then
    /usr/bin/python3.12 "$recipe/scripts/download.py" "$PWD/model"
fi
if [[ $scope == model ]]; then
    exit 0
fi

export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_DOWNLOADS=never
if [[ ! -e .venv/bin/python ]]; then
    uv venv --python /usr/bin/python3.12 .venv
fi
uv pip sync --python .venv/bin/python --default-index https://pypi.org/simple \
    --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
    requirements.txt

if [[ ! -f /usr/include/python3.12/Python.h ]]; then
    mkdir -p .runtime/headers
    (
        cd .runtime/headers
        version=$(dpkg-query -W -f='${Version}' python3.12)
        apt-get download "libpython3.12-dev=$version"
        dpkg-deb -x libpython3.12-dev_*.deb .
    )
fi

site=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
cp -a --remove-destination "$recipe/overlay/vllm/." "$site/vllm/"
fast=$(mktemp -d "$PWD/.runtime/fast.XXXXXXXX")
trap 'rm -rf -- "$fast"' EXIT
cp -a "$site/vllm" "$fast/vllm"
patch --directory "$fast" --strip 1 --batch --dry-run < patches/fast.patch
patch --directory "$fast" --strip 1 --batch < patches/fast.patch
rm -rf -- .runtime/fast
mv -T "$fast" .runtime/fast
trap - EXIT
# NVIDIA labels the ARM64 cuSPARSELt wheel as SBSA in its internal metadata.
if ! report=$(NO_COLOR=1 uv pip check --python .venv/bin/python 2>&1); then
    printf '%s\n' "$report" >&2
    if ! grep -Fxq 'Found 1 incompatibility' <<< "$report" ||
        ! grep -Fxq "The package \`nvidia-cusparselt-cu13\` was built for a different platform" <<< "$report"; then
        exit 1
    fi
    .venv/bin/python - <<'PYTHON'
from ctypes import CDLL
from pathlib import Path
from sysconfig import get_path

library = Path(get_path("purelib")) / "nvidia/cusparselt/lib/libcusparseLt.so.0"
CDLL(str(library))
print("The ARM64 cuSPARSELt library loads.")
PYTHON
else
    printf '%s\n' "$report"
fi
