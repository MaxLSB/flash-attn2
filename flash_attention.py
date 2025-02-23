import torch
import triton
import triton.language as tl

# Idea: Adapt Flash Attention for Sliding Window or for another case.
# This implementation focus on a causal attention only.


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal):
        # Note: Q, K and V are the matrices after the linear layer in the attention.
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        HEAD_DIM_Q = Q.shape[-1]
        HEAD_DIM_K = K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        factor = 1 / HEAD_DIM**0.5

        # We make sure the HEAD_DIM for Q, K and V are the same.
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        # Tensor where we will store the output
        O = torch.empty_like(Q)
        stage = 3 if causal else 1  # Causal or not causal

        # First Dim (X): Which group of queries are we going to work with. How many blocks of Q we have.
        # Second Dim (Y): Which head of which batch element we are going to work with
        # Third Dim (Z): We set it to 1 because we don't want to use it
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # Number of parallel programs (kernels): (BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q)

        # L is the logsumexp for the backward pass, one for each query
        L = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        # Strides for the dimension of Q, K, V, O tensors are the same so we only pass them once (because its causal attention)
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            factor=factor,
            L=L,
            O=O,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, 0, M)
        ctx.grid = grid
        ctx.factor = factor
        ctx.causal = causal
        ctx.HEAD_DIM = HEAD_DIM_K

        return 0


# Kernel for the forward pass of Flash Attention
@triton.jit
def _attn_fwd(
    Q,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    K,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    V,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    factor,
    L,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
    O,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    pass
