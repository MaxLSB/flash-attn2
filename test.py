import torch
import triton
import triton.language as tl

####################################### Test #######################################


def test_methods(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):

    # We initialize the Q, K, V vectors from a normal distribution
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

    factor = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)  # for the backward pass

    # Basic Pytorch Attention implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN)))  # (seq_len, seq_len)
    P = (
        torch.mm(Q, K.transpose(-1, -2)) * factor
    )  # (batch_size, num_heads, seq_len, seq_len)

    if causal:
        # It's causal so the upper triangle of the matrix is filled with infinite values
        P[:, :, MASK == 0] = float("-inf")

    P = torch.softmax(P.float(), dim=-1)
    basic_out = torch.mm(P, V)

    basic_out.backward(dO)
    basic_dQ, Q.grad = Q.grad.clone(), None
    basic_dK, K.grad = K.grad.clone(), None
    basic_dV, V.grad = V.grad.clone(), None

    # Flash Attention implementation in Triton
    flash_out = FlashAttention.apply(Q, K, V, causal).half()
    flash_out.backward(dO)
    flash_dQ, Q.grad = Q.grad.clone(), None
    flash_dK, K.grad = K.grad.clone(), None
    flash_dV, V.grad = V.grad.clone(), None

    # Compare the two implementations and make sure the results match
    assert torch.allclose(basic_out, flash_out, rtol=0.0, atol=1e-2)
    assert torch.allclose(basic_dQ, flash_dQ, rtol=0.0, atol=1e-2)
    assert torch.allclose(basic_dK, flash_dK, rtol=0.0, atol=1e-2)
    assert torch.allclose(basic_dV, flash_dV, rtol=0.0, atol=1e-2)
