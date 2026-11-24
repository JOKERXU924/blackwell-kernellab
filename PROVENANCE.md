# Source provenance

- `kernel.py` is the kernel and tensor launcher extracted from the frozen L2 implementation in Blackwell-KernelLab. The engine-specific registration wrapper is excluded. The SHA-256 of the complete frozen source module before extraction is `ae83700b5ebdebd055be92cca476351461e4743e94a2a54477bedc42d07b0b94`.
- The validation harness uses locally generated synthetic tensors. It is a standalone check, not a substitute for the internal model, real-input or benchmark harness.
- No engine source, model weights, request fixtures, raw traces, environments, or deployment data are vendored. PyTorch and Triton are external dependencies governed by their own terms.
- This repository does not specify an open-source license.
