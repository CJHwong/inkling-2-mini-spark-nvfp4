# Inkling 2 Mini on DGX Spark

Serve Inkling 2 Mini on one NVIDIA DGX Spark with native vLLM and `uv`.

- [Tutorial](#tutorial) gets your first response.
- [How-to guides](#how-to-guides) cover preset changes, Pi, and benchmarks.
- [Reference](#reference) lists requirements, settings, and measured results.
- [Explanation](#explanation) describes why the presets behave differently.

## Tutorial

### Get your first response

1. Open a terminal on the Spark. Go to the clone and check the
   [system requirements](#system-requirements).
2. Install the runtime and download the checkpoint.

   ```bash
   ./setup.sh
   ```

   Setup verifies 13 model files. The download takes about 20 GB of disk space.
3. Start the server.

   ```bash
   ./serve.sh fast
   ```

   Leave this terminal open. The first start can take several minutes
   while the runtime compiles kernels.
4. Open a second terminal on the Spark. Check that the server is ready.

   ```bash
   curl -fsS http://127.0.0.1:8020/health
   ```

   A successful request returns HTTP 200.
5. Send a request through the chat completion endpoint.

   ```bash
   curl -fsS http://127.0.0.1:8020/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "inkling",
       "messages": [{"role": "user", "content": "What is 17 times 24? Give the number and nothing else."}],
       "max_tokens": 512,
       "temperature": 0,
       "reasoning_effort": "low"
     }'
   ```

   Look for `408` in `choices[0].message.content`.
6. Press Ctrl+C in the first terminal to stop the server.

## How-to guides

### Switch presets

1. Stop the current server. Both presets use port `8020`.
2. Start the preset you want.

   ```bash
   ./serve.sh q8
   ```

   Use `./serve.sh fast` to switch back.

### Connect Pi from another machine

1. Start either preset on the Spark.
2. If your SSH alias is `spark`, open a tunnel from the machine running Pi.

   ```bash
   ssh -N -L 8020:127.0.0.1:8020 spark
   ```

3. In Pi, set the OpenAI-compatible provider's base URL to
   `http://127.0.0.1:8020/v1` and the model ID to `inkling`.
   Keep the tunnel open for the session.

### Run the preset benchmark

1. Complete setup and stop any server on port `8020`.
2. Run the benchmark from the clone.

   ```bash
   ./bench.sh
   ```

3. Open the printed directory under `runs/`. It contains results, startup
   times, and server logs for both presets. The script starts and stops
   each preset in order.

## Reference

### System requirements

Use an ARM64 DGX Spark with Ubuntu 24.04, system Python 3.12, and
CUDA 13.0 at `/usr/local/cuda`. Put `uv` on your PATH. Setup uses Git,
curl, patch, and the Ubuntu package tools. Keep 40 GB of free disk
space for the checkpoint, wheels, and kernel builds.

### Commands

| Command | Behavior |
|---|---|
| `./setup.sh` | Fetches the pinned recipe, verifies the model, and installs the runtime. |
| `./setup.sh model` | Downloads and verifies the pinned model. |
| `./setup.sh runtime` | Installs dependencies and both presets. |
| `./serve.sh [q8\|fast]` | Serves the selected preset in the foreground. The default is `fast`. |
| `./bench.sh` | Runs the text benchmark for `q8`, then `fast`. |

Setup checks file sizes and verifies SHA-256 checksums for large model
files. If system Python headers are missing, setup extracts the matching
Ubuntu package inside the clone. If `uv pip check` reports only the
cuSPARSELt platform tag mismatch, setup checks that the library loads.

[Source pins and attribution](NOTICE.md)

### Presets and endpoint

| Preset | Weight paths | Key-value cache |
|---|---|---|
| `q8` | [sxuff's INT8 draft and Triton overlay](https://github.com/sxuff/inkling-2-mini-spark/tree/35a4fd8344f01148d35c899cc107aab0c537ce3d/overlay) | Bfloat16 (BF16) |
| `fast` | NVIDIA FP4 (NVFP4) draft and changes to attention, output head, and expert paths | 8-bit floating point (FP8) |

Both presets serve model `inkling` at `http://127.0.0.1:8020/v1`.
They accept text and set a limit of 262,144 tokens per sequence,
including generated tokens.

### Fresh-clone benchmark

The test used one DGX Spark with Ubuntu 24.04.5, NVIDIA driver
580.178.04, CUDA 13.0.88, and uv 0.13.8. Each preset started with an
empty kernel cache. Startup time ends when `/health` returns HTTP 200.

sxuff's benchmark runs a canary, a warmup, and nine serial requests.
Each timed request generates 160 tokens at low reasoning effort.
Decode speed excludes time to first token. Case medians use tokens per second.

| Preset | Median decode (tokens/s) | Code | Prose | Math | Median time to first token | First startup |
|---|---:|---:|---:|---:|---:|---:|
| [`q8`](results/q8.json) | 67.23 | 73.80 | 45.87 | 67.23 | 163 ms | 206 s |
| [`fast`](results/fast.json) | 85.93 | 90.16 | 52.09 | 85.93 | 171 ms | 325 s |

Both presets passed the canary and completed all nine requests.
They matched 0 of 9 output hashes.

## Explanation

### Why keep two presets?

`q8` provides a baseline with sxuff's draft patch on the same vLLM
build. `fast` changes the draft and target weight paths and uses an
FP8 key-value cache. These changes alter the amount of data the
runtime reads during generation. The benchmark measures their combined
effect.

In the fresh-clone benchmark, `fast` improved median decode speed by
27.8%. Its first startup took longer and included NVFP4 kernel
compilation and tuning. Startup time measures readiness, and decode
speed measures generation after the first token. The measurements
apply to short text prompts with one active request.
