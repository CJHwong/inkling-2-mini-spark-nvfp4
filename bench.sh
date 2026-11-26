#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p runs
run=$(mktemp -d "$PWD/runs/text.XXXXXXXX")
echo "Results: $run"
benchmark="$run/bench.py"
sed 's/^MODEL = "thinkingmachines\/Inkling-2-Mini"$/MODEL = "inkling"/' \
    .runtime/recipe/scripts/bench.py > "$benchmark"

pid=
stop_server() {
    if [[ -n $pid ]]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" || true
        pid=
    fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for preset in q8 fast; do
    if ss -Hlt 'sport = :8020' | grep -q .; then
        echo 'Port 8020 is already in use.' >&2
        exit 1
    fi
    started=$SECONDS
    ./serve.sh "$preset" > "$run/$preset.log" 2>&1 &
    pid=$!
    for ((attempt=0; attempt<600; attempt++)); do
        if curl -fsS --noproxy '*' --max-time 2 http://127.0.0.1:8020/health >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Startup failed. Read $run/$preset.log." >&2
            exit 1
        fi
        sleep 2
    done
    if ((attempt == 600)); then
        echo "Startup timed out. Read $run/$preset.log." >&2
        exit 1
    fi
    printf "%s ready in %s seconds\n" "$preset" "$((SECONDS - started))" | tee -a "$run/startup.txt"
    options=()
    if [[ $preset == fast ]]; then
        options=(--compare "$run/q8.json")
    fi
    .venv/bin/python "$benchmark" "$run/$preset.json" --no-vision "${options[@]}"
    .venv/bin/python - "$run/$preset.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
if result["canary"] != {"http": 200, "answer": "408"}:
    sys.exit(f"Canary failed: {result['canary']!r}")
if len(result["rows"]) != 9 or any(
    row["usage"]["completion_tokens"] != 160 for row in result["rows"]
):
    sys.exit("Expected nine rows of 160 generated tokens each.")
PY
    stop_server
done
