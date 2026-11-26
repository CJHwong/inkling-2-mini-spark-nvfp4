# Third-party notices

The `q8` preset uses sxuff's unmodified DSpark INT8 draft and Triton GEMV
overlay at commit `35a4fd8344f01148d35c899cc107aab0c537ce3d`. Setup also
uses sxuff's checkpoint downloader. The benchmark uses sxuff's test
script with the served model name set to `inkling`.

[sxuff's Inkling 2 Mini Spark recipe and upstream notices](https://github.com/sxuff/inkling-2-mini-spark/blob/35a4fd8344f01148d35c899cc107aab0c537ce3d/NOTICE.md)

`patches/fast.patch` changes the attention projections, output head,
expert path, and draft quantization in vLLM and sxuff's overlay.
vLLM and sxuff's recipe use the Apache License 2.0.

[vLLM source at the pinned commit](https://github.com/vllm-project/vllm/tree/ab18334a4f60da74db7de89fe95594e5ed0cda36)

Setup downloads the Apache-2.0 Inkling 2 Mini checkpoint from Thinking
Machines Lab at revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`.

[Inkling 2 Mini model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7)
