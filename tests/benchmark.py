import torch
import matplotlib.pyplot as plt
import time
import argparse

from src.flash_attention.flash_attention import FlashAttention
from src.multi_head_attention import multi_head_attention

###################################### Benchmark Speed ######################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch vs Triton Attention Speed"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=64, help="Dimension of each attention head"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Window size for sliding window attention",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="global",
        choices=["global", "causal", "sliding_window"],
        help="Attention mode",
    )
    return parser.parse_args()


def benchmark(
    BATCH_SIZE,
    NUM_HEADS,
    HEAD_DIM,
    WINDOW_SIZE,
    attn_mode,
    seq_lens,
    device="cuda",
):
    results = {"torch": [], "triton": []}

    for provider in ["torch", "triton"]:
        for seq_len in seq_lens:
            tflops = compute_flops(
                BATCH_SIZE,
                NUM_HEADS,
                seq_len,
                HEAD_DIM,
                WINDOW_SIZE,
                attn_mode,
                provider,
                device=device,
            )
            results[provider].append(tflops)

    plt.figure(figsize=(8, 6), dpi=100)
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.plot(
        seq_lens,
        results["torch"],
        label="PyTorch Attention",
        linestyle="-",
        markersize=6,
        linewidth=2,
    )
    plt.plot(
        seq_lens,
        results["triton"],
        label="FlashAttention-2 Triton",
        linestyle="--",
        markersize=6,
        linewidth=2,
    )
    plt.title(
        f"{attn_mode.capitalize()} Attention forward + backward speed (RTX 3060)",
        fontsize=12,
        fontweight="bold",
    )
    plt.yticks(fontsize=10)
    plt.xticks([512, 1024, 2048, 4096, 8192], fontsize=10)
    plt.legend(fontsize=12, loc="center right", frameon=True)
    plt.xlabel("Sequence Length", fontsize=12, fontweight="bold")
    plt.ylabel("Speed (TFLOPS)", fontsize=12, fontweight="bold")
    plt.savefig("media/benchmark.png", dpi=300)
    plt.show()


def compute_flops(
    BATCH_SIZE,
    NUM_HEADS,
    SEQ_LEN,
    HEAD_DIM,
    WINDOW_SIZE,
    attn_mode,
    provider,
    device="cuda",
):
    dtype = torch.float16

    Q = torch.randn(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    K = torch.randn(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    V = torch.randn(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    dO = torch.randn_like(Q)

    if provider == "torch":
        fn = lambda: multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode)
    elif provider == "triton":
        fn = lambda: FlashAttention.apply(Q, K, V, WINDOW_SIZE, attn_mode)

    # Forward Pass
    fn()  # Warm-up run
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    fwd_sec = time.perf_counter() - start_time
    flops_per_matmul = 2.0 * BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    fwd_flops = 2 * flops_per_matmul  # Forward pass has 2 matmul

    if attn_mode == "causal":
        fwd_flops *= 0.5  # half the global flops
    elif attn_mode == "sliding_window":
        # An estimation not exactly this
        fwd_flops *= WINDOW_SIZE / SEQ_LEN

    # Backward pass
    O = fn()
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    O.backward(dO, retain_graph=True)
    torch.cuda.synchronize()
    bwd_sec = time.perf_counter() - start_time
    bwd_flops = fwd_flops
    if attn_mode == "causal":
        # 2.0 (bwd) + 0.5 (recompute)
        bwd_flops *= 2.0 + 0.5
    elif attn_mode == "sliding_window":
        # 2.0 (bwd) + WINDOW_SIZE / SEQ_LEN (recompute)
        bwd_flops *= 2.0 + WINDOW_SIZE / SEQ_LEN
    elif attn_mode == "global":
        # 2.0 (bwd)
        bwd_flops *= 2.0

    # TFLOPS
    total_flops = (fwd_flops / fwd_sec + bwd_flops / bwd_sec) * 1e-12

    return total_flops


if __name__ == "__main__":
    args = parse_args()

    # Set the sequence lengths to benchmark
    seq_lens = [512 * i for i in range(1, 14)]

    benchmark(
        BATCH_SIZE=args.batch_size,
        NUM_HEADS=args.num_heads,
        HEAD_DIM=args.head_dim,
        WINDOW_SIZE=args.window_size,
        attn_mode=args.attn_mode,
        seq_lens=seq_lens,
        device="cuda",
    )
