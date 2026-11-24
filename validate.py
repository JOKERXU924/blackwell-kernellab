import argparse
import json
from pathlib import Path

import torch


WIDTH = 12288
ROWS = 4096
BATCHES = range(2, 9)


def ordered_reference(source, weight):
    if any(tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16
           or tensor.ndim != 2 or not tensor.shape[0] or tensor.shape[1] != WIDTH
           for tensor in (source, weight)):
        raise ValueError("The witness reference requires nonempty CPU BF16 matrices of width 12288")
    source, weight = source.float(), weight.float()
    accum = torch.zeros((source.shape[0], weight.shape[0], 32), dtype=torch.float32)
    for start in range(0, WIDTH, 32):
        columns = slice(start, start + 32)
        accum = accum + source[:, None, columns] * weight[None, :, columns]
    for stride in (16, 8, 4, 2, 1):
        accum = accum[..., :stride] + accum[..., stride:2 * stride]
    return accum.squeeze(-1)


def witness():
    source = torch.ones((8, WIDTH), dtype=torch.bfloat16)
    changed = source.clone()
    changed[:, 96] = -1
    weight = torch.zeros((3, WIDTH), dtype=torch.bfloat16)
    patterns = ({0: 2**24, 32: 1, 64: -(2**24), 96: 1, 128: 2**-8},
                {96: 1, 128: 2**-8, 160: 2**-23},
                {0: 2**24, 16: 1, 8: -(2**24), 4: 1, 36: 2**-8})
    for row, pattern in enumerate(patterns):
        for column, value in pattern.items():
            weight[row, column] = value
    return source, changed, weight


def require_exact(actual, expected, label):
    if (actual.shape, actual.dtype, actual.device) != (expected.shape, expected.dtype, expected.device):
        raise AssertionError(f"{label}: shape, dtype or device differs")
    first, second = (tensor.contiguous().view(torch.uint8) for tensor in (actual, expected))
    if not torch.equal(first, second):
        raise AssertionError(f"{label}: {int((first != second).sum())} differing bytes")


def validate_cpu():
    source, changed, weight = witness()
    original = ordered_reference(source, weight)
    alternative = ordered_reference(changed, weight)
    expected_original = torch.tensor([1 + 2**-8, 1 + 2**-8 + 2**-23, 1 + 2**-8])
    expected_changed = torch.tensor([-1 + 2**-8, -1 + 2**-8 + 2**-23, 1 + 2**-8])
    require_exact(original, expected_original.repeat(8, 1), "original FP32 witness")
    require_exact(alternative, expected_changed.repeat(8, 1), "changed FP32 witness")
    require_exact(original[0, :2].bfloat16(), torch.tensor([1, 1 + 2**-7]).bfloat16(), "BF16 midpoint")
    wrong_order = (source[:, None, :].double() * weight[None, :, :].double()).sum(-1).float()
    rejected = int((original != wrong_order).sum())
    if rejected != 16:
        raise AssertionError("The synthetic witnesses did not reject reordered accumulation")
    return dict(witness_outputs=original.numel(), reordered_outputs_rejected=rejected)


def validate_cuda(expected_uuid, result):
    from kernel import applicable, project

    if torch.cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one idle SM120 GPU before launch")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 0):
        raise ValueError("This preview is validated only on SM120")
    if expected_uuid and str(props.uuid).removeprefix("GPU-") != expected_uuid.removeprefix("GPU-"):
        raise ValueError("Visible CUDA device does not match the requested UUID")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    generator = torch.Generator(device="cuda").manual_seed(5090)
    source = torch.randn((8, WIDTH), device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn((ROWS, WIDTH), device="cuda", dtype=torch.bfloat16, generator=generator)
    original, changed, prefix = witness()
    witness_weight = weight.clone()
    witness_weight[:len(prefix)].copy_(prefix)
    if applicable(source[:1], weight):
        raise AssertionError("Unsupported B1 was accepted")
    result["batches"] = []
    for batch in BATCHES:
        data = source[:batch].contiguous()
        native = torch.cat([torch.nn.functional.linear(row, weight) for row in data.split(1)])
        eager = project(data, weight)
        require_exact(eager, native, f"B{batch} native rowwise BF16")
        static = original[:batch].cuda()
        eager32 = project(static, witness_weight, fp32=True)
        eager16 = project(static, witness_weight)
        expected = ordered_reference(original[:batch], prefix).cuda()
        require_exact(eager32[:, :len(prefix)], expected, f"B{batch} FP32 witness")
        require_exact(eager16, eager32.bfloat16(), f"B{batch} BF16 output rounding")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured32 = project(static, witness_weight, fp32=True)
            captured16 = project(static, witness_weight)
        for state, fixture in (("original", original), ("changed", changed), ("restored", original)):
            static.copy_(fixture[:batch])
            graph.replay()
            expected32, expected16 = project(static, witness_weight, fp32=True), project(static, witness_weight)
            require_exact(captured32, expected32, f"B{batch} {state} Graph FP32")
            require_exact(captured16, expected16, f"B{batch} {state} Graph BF16")
            expected_prefix = ordered_reference(fixture[:batch], prefix).cuda()
            require_exact(captured32[:, :len(prefix)], expected_prefix, f"B{batch} {state} Graph witness")
        result["batches"].append(batch)
    result["GPU"] = props.name


def main():
    parser = argparse.ArgumentParser(description="Validate the frozen L2 projection with synthetic tensors")
    parser.add_argument("mode", choices=("cpu", "cuda"))
    parser.add_argument("--expected-uuid")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stream = args.output.open("x", encoding="utf-8") if args.output else None
    result = dict(mode=args.mode, status="running")
    try:
        result.update(validate_cpu())
        if args.mode == "cuda":
            validate_cuda(args.expected_uuid, result)
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        print(json.dumps(result, sort_keys=True))
        if stream is not None:
            with stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")


if __name__ == "__main__":
    main()
