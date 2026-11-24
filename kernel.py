import torch
import triton
import triton.language as tl


@triton.jit
def _multirow_down(Source, Weight, Output, Batch: tl.constexpr, Rows: tl.constexpr, Width: tl.constexpr):
    batches = tl.program_id(1) * 4 + tl.arange(0, 4)
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    lanes = tl.arange(0, 32)
    accumulator = tl.full((4, 4, 32), 0, tl.float32)
    for chunk in range(triton.cdiv(Width, 32 * 16)):
        for item in tl.static_range(16):
            columns = chunk * (32 * 16) + item * 32 + lanes
            source = tl.load(Source + batches[:, None] * Width + columns[None, :],
                (batches[:, None] < Batch) & (columns[None, :] < Width), other=0).to(tl.float32)
            weight = tl.load(Weight + rows[:, None] * Width + columns[None, :],
                (rows[:, None] < Rows) & (columns[None, :] < Width), other=0,
                eviction_policy="evict_first").to(tl.float32)
            accumulator = tl.fma(weight[None, :, :], source[:, None, :], accumulator)
    result = tl.sum(accumulator, 2)
    tl.store(Output + batches[:, None] * Rows + rows[None, :], result,
        (batches[:, None] < Batch) & (rows[None, :] < Rows))


def applicable(source, weight):
    return (source.ndim == 2 and 2 <= source.shape[0] <= 8 and source.shape[1] == 12288
            and weight.shape == (4096, 12288) and source.dtype == weight.dtype == torch.bfloat16
            and source.is_cuda and source.device == weight.device
            and source.is_contiguous() and weight.is_contiguous())


def project(source, weight, fp32=False):
    if not applicable(source, weight):
        raise ValueError("Frozen BF16 multirow Qwen3 down projection required")
    batch = source.shape[0]
    output = torch.empty((batch, weight.shape[0]), device=source.device,
                         dtype=torch.float32 if fp32 else torch.bfloat16)
    _multirow_down[(triton.cdiv(weight.shape[0], 4), triton.cdiv(batch, 4))](
        source, weight, output, batch, *weight.shape, num_warps=4, enable_fp_fusion=False)
    return output
