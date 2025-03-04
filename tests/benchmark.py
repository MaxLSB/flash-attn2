import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from src.flash_attention.flash_attention import FlashAttention
from src.multi_head_attention import multi_head_attention


###################################### Benchmark Runtime ######################################


def benchmark(
    BATCH_SIZE,
    NUM_HEADS,
    HEAD_DIM,
    WINDOW_SIZE,
    attn_mode,
    seq_lens,
    dtype=torch.float16,
):
    """
    Benchmark PyTorch and Triton implementations across different sequence lengths.
    """
    torch_times = []
    triton_times = []

    for SEQ_LEN in seq_lens:
        Q = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        ).normal_(mean=0.0, std=0.5)
        K = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        ).normal_(mean=0.0, std=0.5)
        V = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        ).normal_(mean=0.0, std=0.5)

        # Warm-up run to get rid of PyTorch overhead
        multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode)
        torch.cuda.synchronize()

        start_py = time.time()
        multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode)
        torch.cuda.synchronize()
        torch_times.append(time.time() - start_py)

        # Warm-up run to get rid of Triton overhead
        FlashAttention.apply(Q, K, V, WINDOW_SIZE, attn_mode)
        torch.cuda.synchronize()

        start_tri = time.time()
        FlashAttention.apply(Q, K, V, WINDOW_SIZE, attn_mode).half()
        torch.cuda.synchronize()
        triton_times.append(time.time() - start_tri)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(seq_lens, np.array(torch_times) * 1000 + 1e-2, label="PyTorch", marker="o")
    plt.plot(seq_lens, np.array(triton_times) * 1000 + 1e-2, label="Triton", marker="s")
    plt.xlabel("Sequence Length")
    plt.ylabel("Runtime (ms)")
    plt.title("PyTorch vs Triton Attention Runtime (Forward Pass)")
    plt.legend()
    plt.yscale("log")
    plt.xticks([512, 1024, 2048, 4096, 8192])
    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    seq_lens = [128, 512, 1024, 2048, 4096, 8192]
    benchmark(
        BATCH_SIZE=8,
        NUM_HEADS=4,
        HEAD_DIM=64,
        WINDOW_SIZE=None,
        attn_mode="global",
        seq_lens=seq_lens,
    )
