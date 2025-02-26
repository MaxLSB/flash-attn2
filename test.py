import torch
import triton
import triton.language as tl

from flash_attention import FlashAttention
from multi_head_attention import MultiHeadAttention

###################################### Test Function ######################################


def test(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    """
    This function verifies that both implementations produce the same output.
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

    # Vanilla Multi-Head Attention in PyTorch
    attn_out, _ = MultiHeadAttention.forward(Q, K, V, causal)
    attn_out.backward(dO)
    attn_dQ, Q.grad = Q.grad.clone(), None
    attn_dK, K.grad = K.grad.clone(), None
    attn_dV, V.grad = V.grad.clone(), None

    # Flash Attention implementation in Triton
    # When using torch.autograd.Function, apply triggers the forward pass
    flash_out = FlashAttention.apply(Q, K, V, causal).half()
    flash_out.backward(dO)
    flash_dQ, Q.grad = Q.grad.clone(), None
    flash_dK, K.grad = K.grad.clone(), None
    flash_dV, V.grad = V.grad.clone(), None

    # Compare the two implementations and make sure the results match
    assert torch.allclose(attn_out, flash_out, rtol=0.0, atol=1e-2)
    assert torch.allclose(attn_dQ, flash_dQ, rtol=0.0, atol=1e-2)
    assert torch.allclose(attn_dK, flash_dK, rtol=0.0, atol=1e-2)
    assert torch.allclose(attn_dV, flash_dV, rtol=0.0, atol=1e-2)


if __name__ == "__main__":
    test(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    test(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=False)
    print("PASSED")
