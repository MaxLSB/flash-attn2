import torch

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
    attn_out = multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode)
    attn_out.backward(dO)
    attn_dV, V.grad = V.grad.clone(), None
    attn_dK, K.grad = K.grad.clone(), None
    attn_dQ, Q.grad = Q.grad.clone(), None

    # Flash Attention 2 implementation in Triton
    flash_out = FlashAttention.apply(Q, K, V, WINDOW_SIZE, attn_mode).half()
    flash_out.backward(dO)
    flash_dV, V.grad = V.grad.clone(), None
    flash_dK, K.grad = K.grad.clone(), None
    flash_dQ, Q.grad = Q.grad.clone(), None

    print(attn_out)
    print(flash_out)

    elementwise_distance(attn_out, flash_out)
    count_large_differences(attn_out, flash_out)
    print_max_difference(attn_out, flash_out)
    
    # Compare the two implementations and make sure the results match
    assert torch.allclose(attn_out, flash_out, rtol=0.0, atol=1e-2)
    # assert torch.allclose(attn_dV, flash_dV, rtol=0.0, atol=1e-2)
    # assert torch.allclose(attn_dK, flash_dK, rtol=0.0, atol=1e-2)
    # assert torch.allclose(attn_dQ, flash_dQ, rtol=0.0, atol=1e-2)


def elementwise_distance(tensor1, tensor2):
    distance = torch.abs(tensor1 - tensor2).mean()
    print(f'Element Wise Distance: {distance:4f}')

def count_large_differences(tensor1, tensor2, threshold=1e-2):
    count = (torch.abs(tensor1 - tensor2) > threshold).sum().item()
    print(f"Number of elements differing by more than {threshold}: {count}")
    
def print_max_difference(tensor1, tensor2):
    max_diff = torch.abs(tensor1 - tensor2).max().item()
    print(f"Largest absolute difference: {max_diff}")

if __name__ == "__main__":
    # SEQ_LEN >= 64 | HEAD_DIM >= 64
    # attn_mode: "global" or "causal" or "sliding_window"
    test(
        BATCH_SIZE=1,
        NUM_HEADS=1,
        SEQ_LEN=128,
        HEAD_DIM=64,
        WINDOW_SIZE=40,
        attn_mode="sliding_window",
    )
    print("\nAll Tests: PASSED!\n")

