# Inkling 2 Mini on DGX Spark

Run two native vLLM presets for Inkling 2 Mini on one DGX Spark. Setup
uses `uv` and downloads the pinned model to the clone. Both presets
serve model `inkling` on `127.0.0.1:8020`. Run one preset at a time.

| Preset | Draft and target paths | Key-value cache |
|---|---|---|
| `q8` | sxuff's INT8 draft and Triton overlay | BF16 |
| `fast` | NVFP4 draft plus changes to attention, output head, and expert paths | FP8 |

Both presets are text-only. They set a maximum sequence length of
262,144 tokens, including generated tokens. Use `q8` to compare against
the upstream overlay on the same native runtime.

## Setup

Use a DGX Spark with Ubuntu 24.04, CUDA 13.0 at `/usr/local/cuda`, and
system Python 3.12. Put `uv` on your PATH. Setup uses Git, curl, patch,
and the Ubuntu package tools. Keep at least 40 GB of free disk space
for the checkpoint, wheel cache, and kernel builds.

From the clone, run:

```bash
./setup.sh
./serve.sh fast
```

Setup verifies the checkpoint file sizes and SHA-256 hashes for large
files. If system Python headers are missing, setup extracts the
matching Ubuntu header package inside the clone.

To change presets, stop the server and run `./serve.sh q8`. Pi can use
either preset with the same OpenAI-compatible endpoint and model name.
If Pi runs on another machine, open a tunnel:

```bash
ssh -N -L 8020:127.0.0.1:8020 spark
```

Use `http://127.0.0.1:8020/v1` as the base URL in Pi.

## Benchmark

With port 8020 free, run `./bench.sh`. It starts and stops each preset
in order. It runs sxuff's canary, warmup, and nine serial requests.
Each timed request generates 160 tokens with low reasoning effort.
The reported decode rates exclude time to first token.

See [NOTICE.md](NOTICE.md) for attribution and the pinned sources.
