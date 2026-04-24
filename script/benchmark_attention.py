#!/usr/bin/env python3
"""Simple attention benchmark for Torch SDPA and xFormers on CUDA."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterator

import torch


@dataclass
class BenchResult:
    name: str
    dtype: str
    batch_size: int
    heads: int
    seq_len: int
    head_dim: int
    causal: bool
    warmup_iters: int
    iters: int
    avg_ms: float
    tokens_per_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark attention kernels (torch SDPA, optional xFormers)."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--sdpa-backend",
        choices=["auto", "flash", "mem_efficient", "math"],
        default="auto",
    )
    parser.add_argument(
        "--bench-xformers",
        action="store_true",
        help="Also benchmark xformers.ops.memory_efficient_attention.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def maybe_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(
    name: str, fn: Callable[[], torch.Tensor], args: argparse.Namespace
) -> BenchResult:
    for _ in range(args.warmup_iters):
        _ = fn()
    maybe_sync()

    start = time.perf_counter()
    for _ in range(args.iters):
        _ = fn()
    maybe_sync()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed * 1000.0) / args.iters
    tokens = args.batch_size * args.seq_len
    tokens_per_sec = tokens / (avg_ms / 1000.0)
    return BenchResult(
        name=name,
        dtype=args.dtype,
        batch_size=args.batch_size,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        causal=args.causal,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        avg_ms=avg_ms,
        tokens_per_sec=tokens_per_sec,
    )


def sdpa_context(backend: str) -> contextlib.AbstractContextManager[object]:
    if backend == "auto":
        return contextlib.nullcontext()

    # torch 2.5+ preferred API
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        mapping = {
            "flash": [SDPBackend.FLASH_ATTENTION],
            "mem_efficient": [SDPBackend.EFFICIENT_ATTENTION],
            "math": [SDPBackend.MATH],
        }
        return sdpa_kernel(mapping[backend])
    except Exception:
        pass

    # fallback for older API
    mapping = {
        "flash": dict(enable_flash=True, enable_mem_efficient=False, enable_math=False),
        "mem_efficient": dict(
            enable_flash=False, enable_mem_efficient=True, enable_math=False
        ),
        "math": dict(enable_flash=False, enable_mem_efficient=False, enable_math=True),
    }
    return legacy_sdpa_context(**mapping[backend])


@contextlib.contextmanager
def legacy_sdpa_context(
    *,
    enable_flash: bool,
    enable_mem_efficient: bool,
    enable_math: bool,
) -> Iterator[None]:
    prev_flash = torch.backends.cuda.flash_sdp_enabled()
    prev_mem_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
    prev_math = torch.backends.cuda.math_sdp_enabled()
    prev_cudnn = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        torch.backends.cuda.enable_flash_sdp(enable_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient)
        torch.backends.cuda.enable_math_sdp(enable_math)
        torch.backends.cuda.enable_cudnn_sdp(False)
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(prev_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(prev_mem_efficient)
        torch.backends.cuda.enable_math_sdp(prev_math)
        torch.backends.cuda.enable_cudnn_sdp(prev_cudnn)


def print_env() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch.cuda: {torch.version.cuda or 'unavailable'}")  # pyright: ignore[reportAttributeAccessIssue]
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")
        print(f"cudnn: {torch.backends.cudnn.version()}")
    try:
        import xformers  # noqa: F401

        print(f"xformers: {xformers.__version__}")
    except Exception as exc:
        print(f"xformers: unavailable ({exc!r})")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for this benchmark.")

    print_env()
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda")

    q = torch.randn(
        args.batch_size,
        args.heads,
        args.seq_len,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with sdpa_context(args.sdpa_backend):
        sdpa_result = benchmark(
            f"torch.sdpa[{args.sdpa_backend}]",
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=args.causal
            ),
            args,
        )

    results: list[BenchResult] = [sdpa_result]

    if args.bench_xformers:
        try:
            import xformers.ops as xops

            q_x = q.permute(0, 2, 1, 3).contiguous()
            k_x = k.permute(0, 2, 1, 3).contiguous()
            v_x = v.permute(0, 2, 1, 3).contiguous()
            attn_bias = xops.LowerTriangularMask() if args.causal else None

            xformers_result = benchmark(
                "xformers.memory_efficient_attention",
                lambda: xops.memory_efficient_attention(
                    q_x, k_x, v_x, attn_bias=attn_bias
                ),
                args,
            )
            results.append(xformers_result)
        except Exception as exc:
            print(f"[warn] xformers benchmark skipped: {exc!r}")

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return

    print("\n=== Results ===")
    for r in results:
        print(
            f"{r.name}: avg={r.avg_ms:.3f} ms/iter, "
            f"throughput={r.tokens_per_sec:,.0f} tokens/s "
            f"(B={r.batch_size}, H={r.heads}, T={r.seq_len}, D={r.head_dim}, dtype={r.dtype}, causal={r.causal})"
        )


if __name__ == "__main__":
    main()
