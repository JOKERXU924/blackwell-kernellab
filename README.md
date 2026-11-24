# Blackwell-KernelLab

A focused Triton implementation of the BF16 MLP down projection for small-batch Qwen3-8B decode on RTX 5090 (SM120). This is a standalone research preview of the L2 kernel, not a drop-in Nano-vLLM patch or a general GEMM replacement.

## What changed

The internal L1 compatibility path evaluates the projection one row at a time. L2 shares each tile's weight loads across up to four input rows. A program computes four input rows by four output channels, retaining the 32-lane serial FP32 FMA order, the final lane reduction and a BF16 store. This changes weight reuse, not the order of an output's dot product.

The exported kernel supports contiguous, same-device BF16 input `[B, 12288]` and weight `[4096, 12288]` for `2 <= B <= 8`. It does not implement bias, tensor parallelism or a serving backend. `fp32=True` exposes the value before BF16 conversion for validation.

## Run the standalone checks

Verified on Linux x86-64, Python 3.12, PyTorch 2.8.0 with CUDA 12.8, Triton 3.4.0 and an RTX 5090. Use an isolated environment. No model download is needed. Set `IDLE_GPU_UUID` to the UUID of an idle SM120 device before the CUDA command.

```sh
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES="" python validate.py cpu
CUDA_VISIBLE_DEVICES="$IDLE_GPU_UUID" python validate.py cuda --expected-uuid "$IDLE_GPU_UUID"
```

The CUDA check requires exactly one visible SM120 GPU. It compares all BF16 outputs with per-row native linear, checks FP32 reduction and BF16 rounding witnesses, and replays both outputs after changing and restoring the captured input at the same address. All inputs are generated locally. An optional `--output results/run-001.json` saves the completed batches or failure without overwriting an existing file; create the parent directory first.

## Full-model result

In a separate Qwen3-8B integration on an RTX 5090, throughput over seven fixed workloads increased **88.29 → 99.76 tokens/s (+13.00%)** against the project's own L1 baseline. The equally weighted geometric mean was **+9.73%**; short-input B8 was **+24.93%**. B1 and time to first token were essentially unchanged.

See [the measurement protocol and limitations](RESULTS.md). The synthetic validation above does **not** rerun the full-model result. The complete engine, model weights, prompts, activation fixtures, and raw run artifacts are not included.

See [PROVENANCE.md](PROVENANCE.md) for the frozen source identity and publication scope.
