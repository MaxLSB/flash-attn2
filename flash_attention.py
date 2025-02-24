import torch
import triton
import triton.language as tl

# Idea: Adapt Flash Attention for Sliding Window or for another case.
# This implementation focus on a causal attention ?


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


# Triton kernel for the forward pass of Flash Attention
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
    # Index of the block in the sequence length to process
    block_index_q = tl.program_id(0)
    # Index representing the combination of batch and head to process. Each program handles one head in one batch.
    index_batch_head = tl.program_id(1)
    # Extract the batch index from the combined batch-head index (assuming each batch contains NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # Extract the head index within the current batch
    index_head = index_batch_head % NUM_HEADS

    # Offset to get the (SEQ_LEN, HEAD_DIM) block in Q, K, V with the index batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_batch + index_head.to(tl.int64) * stride_head
    )

    # Note: Q, K, V are pointers
    # Pointer to the current Q block: Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # Note: Each program iterate through all the keys and values so the offsets are (0, 0).

    # Pointer to the current V block: V[index_batch, index_head, :, :]
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    # Pointer to the current K block: K[index_batch, index_head, :, :]
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_dim,
            stride_seq,
        ),  # We invert the strides to transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    # Pointer to the current O block: O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # offsets for the tokens in Q to process
    query_offsets = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # offsets for the tokens in K and V sequence to process
    key_values_offsets = tl.arange(0, BLOCK_SIZE_KV)

    # maximum for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

    # sum for each query (we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # accumulator for the output. A group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the Q block in SRAM
    Q_block = tl.load(Q_block_ptr)

    # STAGE = 3 if causal | STAGE = 1 if non-causal
    if STAGE == 1 or STAGE == 3:
        # For non-causal attention: This steps runs all the blocks.
        # For causal attention: This steps runs the blocks to the left of the diagonal
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            query_offsets,
            key_values_offsets,
            SEQ_LEN,
        )

    if STAGE == 3:
        # For causal attention only: This step runs for the blocks in which there is transition between nn-masked and masked keys
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            query_offsets,
            key_values_offsets,
            SEQ_LEN,
        )


# Triton kernel for the inner loop of the forward pass of Flash Attention
@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    factor,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    query_offsets: tl.constexpr,
    key_values_offsets: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN
