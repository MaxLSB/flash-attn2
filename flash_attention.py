import torch
import triton
import triton.language as tl

# Self-Attention only with both the causal and non-causal cases.
# Flash-bidirectional-attention & Flash-causal-attention
# Change the STAGE system
# Flash Attention for Sliding Window
# Multi-Head Latent Flash Attention
# GQA Flash Attention

############################### Flash Attention Forward/Backward pass ###############################


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal):
        # Note: Q, K and V are the matrices after the linear layer in the attention.
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        HEAD_DIM_Q = Q.shape[-1]
        HEAD_DIM_K = K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        softmax_factor = 1 / HEAD_DIM**0.5

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
            softmax_factor=softmax_factor,
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

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.grid = grid
        ctx.softmax_factor = softmax_factor
        ctx.causal = causal
        ctx.HEAD_DIM = HEAD_DIM_K

        return 0


# Triton kernel for the forward pass of Flash Attention
@triton.jit
def _attn_fwd(
    Q,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    K,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    V,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    softmax_factor,
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

    # array of indices which represent the offsets for the tokens in Q to process
    query_offsets = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # array of indices which represent the offsets for the tokens in K and V sequence to process
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
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            query_offsets,
            key_values_offsets,
            SEQ_LEN,
        )

    # The reason we don't fuse the for loop to the left of the diagonal with the one exactly on the diagonal for the causal attention is to optimize the pipelining that Triton does.
    if STAGE == 3:
        # For causal attention only: This step runs for the blocks in the diagonal, in which there is transition between nn-masked and masked keys
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            query_offsets,
            key_values_offsets,
            SEQ_LEN,
        )

        # Broadcasting so it's done element wise just like with a diag matrix
        O_block = O_block / l_i[:, None]

        # Pointer to the correct L block. We need to skip by SEQ_LEN for each head in each batch and by query_offsets to select the correct tokens for the current query block.
        L_block_ptr = L + index_batch_head * SEQ_LEN + query_offsets

        # logsumexp for the backward pass
        tl.store(L_block_ptr, m_i + tl.math.log(l_i))
        # Save O_block to the correct location pointed to by O_block_ptr
        tl.store(O_block_ptr, O_block.to(O.type.element_ty))


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
    softmax_factor,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    query_offsets: tl.constexpr,
    key_values_offsets: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # lower and higher index of the key blocks that we should work with
    if STAGE == 1:
        # Only for causal attention, we consider the key blocks on the left of the diagonal
        lower, higher = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Only for causal attention, we consider the key blocks in which there is transition between non-masked and masked keys (so in the diagonal)
        lower, higher = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lower = tl.multiple_of(lower, BLOCK_SIZE_Q)
    else:
        # Only for non-causal attention, we consider all the key blocks
        lower, higher = 0, SEQ_LEN

    # Adjust the pointer to the correct block
    K_block_ptr = tl.advance(
        K_block_ptr, (0, lower)
    )  # offset is inverted because K is transposed
    V_block_ptr = tl.advance(V_block_ptr, (lower, 0))

    # LOOP over all the keys and values blocks and store inside the accumulator: O_block

    for start_kv in range(lower, higher, BLOCK_SIZE_KV):
        # Let the compiler know that start_kv is a multiple of BLOCK_SIZE_KV, so the compiler can do optimization
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # load the K block in SRAM and compute the dot product
        K_block = tl.load(K_block_ptr)
        # (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            # Build the causal mask for the QK_block
            mask = query_offsets[:, None] >= (start_kv + key_values_offsets[None, :])
            # # Apply causal mask by adding -1.0e6 to masked positions (upper triangle)
            QK_block = QK_block * softmax_factor + tl.where(mask, 0, -1.0e6)
            # max per row for stability (log-sum-exp trick)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            # subtracting the row-wise max for numerical stability
            QK_block -= m_ij[:, None]
        else:
            # compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_factor)
            QK_block = QK_block * product - m_ij[:, None]

        # exp(qk_ij - m_i)
        P_block = tl.math.exp(QK_block)

        # Correction factor
        cor_fact = tl.math.exp(m_i - m_ij)

        # Apply correction factor to the previous l_i and add the new l_ij
        l_i = cor_fact * l_i + tl.sum(P_block, 1)

        # load the V block on SRAM
        V_block = tl.load(V_block_ptr)

        # Is it really needed ?
        P_block = P_block.to(tl.float16)

        # Compute the new O_block
        O_block = O_block * cor_fact[:, None] + tl.dot(P_block, V_block)

        # update the maximum tensor
        m_i = m_ij

        # Move the pointers to the next K and V blocks
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i
