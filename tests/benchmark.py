import torch
import matplotlib.pyplot as plt
import argparse
import torch.utils.benchmark as benchmark

from src.flash_attention.flash_attention import FlashAttention
from src.multi_head_attention import multi_head_attention

###################################### Benchmark Speed (TFLOPs/s) for Fwd + Bwd Pass ######################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch vs Triton Attention Speed"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=128, help="Dimension of each attention head"
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


def calculate_flops(seq_len, batch_size, num_heads, head_dim, attn_mode, window_size):
    """Calculate floating point operations for attention mechanisms."""
    flops_per_matmul = 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim
    fwd_flops = 2 * flops_per_matmul  # Forward pass has 2 matmul
    bwd_flops = 5 * flops_per_matmul  # Backward pass has 5 matmul

    if attn_mode == "causal":
        # Approximate half the global flops for causal attention
        fwd_flops *= 0.5
        bwd_flops *= 0.5
    elif attn_mode == "sliding_window":
        # Scale by window size ratio for sliding window attention
        ratio = window_size / seq_len
        fwd_flops *= ratio
        bwd_flops *= ratio

    return fwd_flops, bwd_flops


def calculate_efficiency(flops, time):
    """Convert raw FLOPS to TFLOPs/s."""
    return flops / time / 1e12


def benchmark_time_combined(
    fn, *inputs, grad=None, repeats=5, amp_dtype=torch.float16, **kwinputs
):
    """Benchmark forward and backward pass of an arbitrary function using torch.utils.benchmark."""

    def combined_fn(grad, *inputs, **kwinputs):
        # Reset gradients to avoid accumulation
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None

        # Forward pass with AMP if enabled
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            attn_out = fn(*inputs, **kwinputs)

        # Backward pass
        attn_out.backward(grad, retain_graph=True)

    # Warmup run to compile
    combined_fn(grad, *inputs, **kwinputs)
    torch.cuda.synchronize()

    # Timer for combined forward + backward pass
    timer = benchmark.Timer(
        stmt="combined_fn(grad, *inputs, **kwinputs)",
        globals={
            "combined_fn": combined_fn,
            "fn": fn,
            "inputs": inputs,
            "grad": grad,
            "kwinputs": kwinputs,
        },
        num_threads=torch.get_num_threads(),
    )

    # Return the mean time in seconds
    time_result = timer.timeit(repeats)
    return time_result.mean


def run_attention_benchmark(
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    window_size,
    attn_mode,
    provider,
    device="cuda",
):
    """Run benchmark for a specific attention implementation and configuration."""
    dtype = torch.float16

    # Create input tensors
    Q = torch.randn(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    K = torch.randn(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    V = torch.randn(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    dO = torch.randn_like(Q)  # Gradient for backward pass

    # Select appropriate attention implementation
    if provider == "torch":
        fn = lambda: multi_head_attention(Q, K, V, window_size, attn_mode)
    elif provider == "triton":
        fn = lambda: FlashAttention.apply(Q, K, V, window_size, attn_mode)

    # Measure execution time
    time = benchmark_time_combined(fn, grad=dO, repeats=5, amp_dtype=dtype)
    print(
        f"> {provider.capitalize()} Attention | Seq_len={seq_len} | {time*1e3:.4f} ms"
    )

    # Calculate TFLOPs/s
    fwd_flops, bwd_flops = calculate_flops(
        seq_len, batch_size, num_heads, head_dim, attn_mode, window_size
    )
    total_tflops = calculate_efficiency(fwd_flops + bwd_flops, time)

    return total_tflops


def plot_benchmark_results(
    batch_size,
    num_heads,
    head_dim,
    window_size,
    attn_mode,
    seq_lens,
    device="cuda",
):
    """Run benchmarks and plot results for comparison."""
    results = {"torch": [], "triton": []}

    # Collect benchmark results for each provider and sequence length
    for provider in ["torch", "triton"]:
        for seq_len in seq_lens:
            tflops = run_attention_benchmark(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                window_size,
                attn_mode,
                provider,
                device=device,
            )
            results[provider].append(tflops)

    # Create the plot
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

    # Set plot labels and styling
    plt.title(
        f"{attn_mode.capitalize()} attention forward + backward speed (RTX 4060 Laptop GPU)",
        fontsize=12,
        fontweight="bold",
    )
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.legend(fontsize=12, loc="center right", frameon=True)
    plt.xlabel("Sequence Length", fontsize=12, fontweight="bold")
    plt.ylabel("Speed (TFLOPs/s)", fontsize=12, fontweight="bold")

    # Save and display the plot
    plt.savefig("media/benchmark.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    # Set the sequence lengths to benchmark
    seq_lens = [512 * i for i in range(1, 11)]

    plot_benchmark_results(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        window_size=args.window_size,
        attn_mode=args.attn_mode,
        seq_lens=seq_lens,
        device="cuda",
    )
