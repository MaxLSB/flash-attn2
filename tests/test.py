import torch
import time

from src.flash_attention.flash_attention import FlashAttention
from src.multi_head_attention import multi_head_attention

###################################### Test Function ######################################


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
    multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode)
    torch.cuda.synchronize()

    # Forward Pass
    start_py = time.time()
    attn_out = multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode)
    torch.cuda.synchronize()
    time_py = time.time() - start_py

    # Backward Pass
    attn_out.backward(dO)
    attn_dV, V.grad = V.grad.clone(), None
    attn_dK, K.grad = K.grad.clone(), None
    attn_dQ, Q.grad = Q.grad.clone(), None

    # Flash Attention 2 implementation in Triton
    # Warm-up run to get rid of Triton overhead
    FlashAttention.apply(Q, K, V, WINDOW_SIZE, attn_mode)
    torch.cuda.synchronize()

    # Forward Pass
    start_tri = time.time()
    flash_out = FlashAttention.apply(Q, K, V, WINDOW_SIZE, attn_mode).half()
    torch.cuda.synchronize()
    time_tri = time.time() - start_tri

    # Backward Pass
    flash_out.backward(dO)
    flash_dV, V.grad = V.grad.clone(), None
    flash_dK, K.grad = K.grad.clone(), None
    flash_dQ, Q.grad = Q.grad.clone(), None

    # Display Information

    print(f"\n> PyTorch Runtime (Forward Pass): {time_py * 1000:.4f} ms.")
    print(f"> Triton Runtime (Forward Pass): {time_tri * 1000:.4f} ms.")
    # Compare the two implementations and make sure the results match
    assert torch.allclose(attn_out, flash_out, rtol=0.0, atol=1e-2)
    print("\n> Forward pass matches.")
    assert torch.allclose(attn_dV, flash_dV, rtol=0.0, atol=1e-2)
    assert torch.allclose(attn_dK, flash_dK, rtol=0.0, atol=1e-2)
    assert torch.allclose(attn_dQ, flash_dQ, rtol=0.0, atol=1e-2)
    print("> Backward pass matches.")


if __name__ == "__main__":

    # Restrictions for BLOCK_SIZE = 16.

    # For Global/Causal/Sliding Window:
    # --> SEQ_LEN >= 16

    # For Sliding Window:
    # --> SEQ_LEN >= 64
    # --> 2 * BLOCK_SIZE <= WINDOW_SIZE <= SEQ_LEN

    test(
        BATCH_SIZE=8,
        NUM_HEADS=8,
        SEQ_LEN=4096,
        HEAD_DIM=64,
        WINDOW_SIZE=1000,
        attn_mode="sliding_window",
    )
    print("\nAll Tests: PASSED!\n")
