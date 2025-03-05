import torch
import time
import argparse

from src.flash_attention.flash_attention import FlashAttention
from src.multi_head_attention import multi_head_attention

###################################### Test Function ######################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch vs Triton Attention Runtime"
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
    parser.add_argument(
        "--seq_len", type=int, nargs="+", default=4096, help="Sequence lengths"
    )
    return parser.parse_args()


def test(
    BATCH_SIZE,
    NUM_HEADS,
    SEQ_LEN,
    HEAD_DIM,
    WINDOW_SIZE,
    attn_mode,
    dtype=torch.float16,
):
    """
    This function verifies that the Triton and PyTorch implementations produce the same output.
    """

    # We initialize the Q, K, V vectors from a normal distribution
    # Q, K, V already went through the first linear layers
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    dO = torch.randn_like(Q)  # for the backward pass

    # Multi-Head Attention in PyTorch

    # Warm-up run to get rid of potential overhead
    attn_out = multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode)
    attn_out.backward(dO)
    torch.cuda.synchronize()

    start_py = time.perf_counter()
    attn_out = multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode)
    attn_out.backward(dO)
    torch.cuda.synchronize()
    time_py = time.perf_counter() - start_py

    attn_dV, V.grad = V.grad.clone(), None
    attn_dK, K.grad = K.grad.clone(), None
    attn_dQ, Q.grad = Q.grad.clone(), None

    # Flash Attention 2 implementation in Triton

    # Warm-up or compile run to get rid of Triton overhead
    flash_out = FlashAttention.apply(Q, K, V, WINDOW_SIZE, attn_mode)
    flash_out.backward(dO)
    torch.cuda.synchronize()

    start_tri = time.perf_counter()
    flash_out = FlashAttention.apply(Q, K, V, WINDOW_SIZE, attn_mode).half()
    flash_out.backward(dO)
    torch.cuda.synchronize()
    time_tri = time.perf_counter() - start_tri

    flash_dV, V.grad = V.grad.clone(), None
    flash_dK, K.grad = K.grad.clone(), None
    flash_dQ, Q.grad = Q.grad.clone(), None

    # Display Information
    print(f"\n> PyTorch Runtime (Forward + Backward Pass): {time_py * 1000:.4f} ms.")
    print(f"> Triton Runtime (Forward + Backward Pass): {time_tri * 1000:.4f} ms.")

    # Compare the two implementations and make sure the results match
    assert torch.allclose(attn_out, flash_out, rtol=0.0, atol=1e-2)
    print("\n> Forward pass matches.")
    assert torch.allclose(attn_dV, flash_dV, rtol=0.0, atol=1e-2)
    assert torch.allclose(attn_dK, flash_dK, rtol=0.0, atol=1e-2)
    assert torch.allclose(attn_dQ, flash_dQ, rtol=0.0, atol=1e-2)
    print("> Backward pass matches.")

    print("\nAll Tests PASSED!\n")


if __name__ == "__main__":

    # Restrictions for BLOCK_SIZE = 16.

    # For Global/Causal/Sliding Window:
    # --> SEQ_LEN >= 16

    # For Sliding Window:
    # --> SEQ_LEN >= 64
    # --> 2 * BLOCK_SIZE <= WINDOW_SIZE <= SEQ_LEN

    args = parse_args()
    test(
        args.batch_size,
        args.num_heads,
        args.seq_len,
        args.head_dim,
        args.window_size,
        args.attn_mode,
    )
