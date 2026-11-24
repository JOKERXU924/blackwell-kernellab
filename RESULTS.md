# Qwen3-8B integration result

RTX 5090, BF16, same-round comparison against the project's frozen **L1** compatibility baseline. These results were recorded on November 18, 2026. They are separate from the standalone synthetic checks in this repository.

| Workload | L1 median tok/s | L2 median tok/s | Throughput change | Inter-token latency change |
| --- | ---: | ---: | ---: | ---: |
| 256 / B1 | 81.24 | 81.17 | −0.08% | +0.05% |
| 512 / B4 | 119.10 | 146.20 | +22.75% | −20.99% |
| 512 / B8 | 124.82 | 155.94 | +24.93% | −22.76% |
| 4096 / B1 | 52.26 | 52.26 | +0.01% | −0.02% |
| 8192 / B1 | 37.36 | 37.33 | −0.08% | +0.06% |
| 4096 / B4 | 68.08 | 76.13 | +11.83% | −20.13% |
| One long + four short | 87.45 | 97.81 | +11.85% | −15.31% |

The run passed 21 real-input operator checks, a 624-position smoke check and the full five-group model regression before collecting performance data. The final regression compared 6,453 prediction positions per execution path with the frozen L1 logits. Both eager and full Decode Graph (including the output head) were bitwise identical to L1 on those inputs. Top-1 mismatches versus the Hugging Face reference were zero under the project's original gates; its logits are not claimed to be bitwise identical.

Timing used separate processes ordered L1 → L2 → L2 → L1. Each process warmed up twice and timed five runs per workload. Both arms handled 240 requests and 15,360 output tokens in the formal trials. Aggregate throughput divides those tokens by the sum of measured durations: 173.98 s for L1 and 153.97 s for L2. The seven-cell geometric mean instead gives each scene's median throughput ratio equal weight.

Identical controlled continuations fixed the work per request. Each request recomputed full Prefill and the measurement retained actual sampling work. This is an engine-level, warmed, closed-workload result: it does not cover HTTP, cold startup, open-loop traffic, or gains versus upstream Nano-vLLM, vLLM, or a native batched GEMM baseline. No uncertainty interval was estimated, and small B1 differences are reported without treating them as significant.
