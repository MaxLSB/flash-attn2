import torch
import triton
import triton.language as tl

# Self-Attention only with both the causal and non-causal cases.
# No dropout in this implementation
# Flash-bidirectional-attention & Flash-causal-attention
# Change the STAGE system
# Flash Attention for Sliding Window
# Multi-Head Latent Flash Attention
# GQA Flash Attention
# Native Sparse Attention with Flash Attention !!

###################################### Flash Attention Class ######################################


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

        # Self-Attention so the strides should be the same
        assert Q.stride() == K.stride() == V.stride() == O.stride()

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
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.grid = grid
        ctx.softmax_factor = softmax_factor
        ctx.causal = causal

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        dQ = torch.empty_like(Q)
        dV = torch.empty_like(V)
        dK = torch.empty_like(K)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        preprocess_grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE"]),
            BATCH_SIZE * NUM_HEADS,
        )

        # (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
        D = torch.empty_like(L)

        # Compute all the D_i elements
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
        )

        # First Dim (X): The number of K, V blocks we will have rounded up
        # Second Dim (Y): Which head of which batch element we are going to work with
        # Third Dim (Z): We set it to 1 because we don't want to use it
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_KV"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        stage = 3 if ctx.causal else 1

        # Unlike the paper, we do 2 for loops, one to compute dK, dV and the other to compute dQ
        # Also, this allows us to only write to the HBM once for each dK, dV, dQ (in the paper they write multiple times to the HBM for the each query blocks, which is slow)

        # K, V blocks are fixed, we itterate through all the Q blocks -> compute dK and dV
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_factor=ctx.softmax_factor,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            L=L,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
        )

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # Q block is fixed, we iterate through all the K, V blocks -> compute dQ
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_factor=ctx.softmax_factor,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            L=L,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
        )

        return dQ, dK, dV, None, None


###################################### Forward Pass Triton kernels ######################################


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
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
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    # Index of the block in the sequence length to process
    block_index_q = tl.program_id(0)
    # Index representing the combination of batch and head to process. Each program handles one head in one batch.
    index_batch_head = tl.program_id(1)
    # Extract the batch index from the combined batch-head index (assuming each batch contains NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # Extract the head index within the current batch
    index_head = index_batch_head % NUM_HEADS

    # Offset to get the (SEQ_LEN, HEAD_DIM) block in Q, K, V with the index batch and head
    qvk_start = (
        index_batch.to(tl.int64) * stride_batch + index_head.to(tl.int64) * stride_head
    )

    # Note: Q, K, V are pointers
    # Pointer to the current Q block: Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_start,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # Note: Each program iterate through all the keys and values so the offsets are (0, 0).

    # Pointer to the current V block: V[index_batch, index_head, :, :]
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_start,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    # Pointer to the current K block: K[index_batch, index_head, :, :]
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_start,
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
        base=O + qvk_start,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # array of indices which represent the offsets for the tokens in Q to process
    q_offsets = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # array of indices which represent the offsets for the tokens in K and V sequence to process
    kv_offsets = tl.arange(0, BLOCK_SIZE_KV)

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
            q_offsets,
            kv_offsets,
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
            q_offsets,
            kv_offsets,
            SEQ_LEN,
        )

        # Broadcasting so it's done element wise just like with a diag matrix
        O_block = O_block / l_i[:, None]

        # Pointer to the correct L block. We need to skip by SEQ_LEN for each head in each batch and by q_offsets to select the correct tokens for the current query block.
        L_block_ptr = L + index_batch_head * SEQ_LEN + q_offsets

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
    q_offsets: tl.constexpr,
    kv_offsets: tl.constexpr,
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
            mask = q_offsets[:, None] >= (start_kv + kv_offsets[None, :])
            # # Apply causal mask by adding -1.0e6 to masked positions (upper triangle)
            QK_block = QK_block * softmax_factor + tl.where(mask, 0, -1.0e6)
            # max per row for stability (log-sum-exp trick)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            # subtracting the row-wise max for numerical stability
            QK_block -= m_ij[:, None]
        else:
            # compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_factor)
            QK_block = QK_block * softmax_factor - m_ij[:, None]

        # exp(qk_ij - m_i)
        P_block = tl.math.exp(QK_block)

        # Correction factor
        cor_fact = tl.math.exp(m_i - m_ij)

        # Apply correction factor to the previous l_i and add the new l_ij
        l_i = cor_fact * l_i + tl.sum(P_block, 1)

        # load the V block on SRAM
        V_block = tl.load(V_block_ptr)

        P_block = P_block.to(tl.float16)

        # Compute the new O_block
        O_block = O_block * cor_fact[:, None]
        # O_block is an accumulator. It is more efficient
        O_block = tl.dot(P_block, V_block, O_block)

        # update the maximum tensor
        m_i = m_ij

        # Move the pointers to the next K and V blocks
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i


###################################### Backward Pass Triton kernels ######################################


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE": BLOCK_SIZE},
        )
        for BLOCK_SIZE in [64, 128]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Index of the block in the sequence length to process
    block_index_q = tl.program_id(0)
    # Index representing the combination of batch and head to process. Each program handles one head in one batch.
    index_batch_head = tl.program_id(1)

    # array of indices which represent the offsets for the tokens in Q to process
    q_offsets = block_index_q * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # array of indices which represent the offsets on dimension. We need to load all the dimensions.
    dim_offsets = tl.arange(0, HEAD_DIM)

    # We load the O_block with the correct pointer (could also be done with tl.make_block_ptr()) -> (BLOCK_SIZE_Q, HEAD_DIM)
    # O: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    O_block = tl.load(
        O
        + index_batch_head * SEQ_LEN * HEAD_DIM
        + q_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )

    # We load a single block -> (BLOCK_SIZE_Q, HEAD_DIM)
    dO_block = tl.load(
        dO
        + index_batch_head * SEQ_LEN * HEAD_DIM
        + q_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    ).to(tl.float32)

    # Compute the D block -> (BLOCK_SIZE_Q,)
    D_block = tl.sum(dO_block * O_block, axis=1)
    D_block_ptrs = D + index_batch_head * SEQ_LEN + q_offsets

    tl.store(D_block_ptrs, D_block)


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_KV in [64, 128]
        for BLOCK_SIZE_Q in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_factor,
    dO,
    dQ,
    dK,
    dV,
    L,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Index representing the combination of batch and head to process
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + index_head * stride_head).to(
        tl.int64
    )
    offset_batch_head_seq = (offset_batch_head * SEQ_LEN).to(tl.int64)

    # put the pointers are the right place for the current batch and head
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dQ += offset_batch_head
    dV += offset_batch_head
    dK += offset_batch_head

    # put the pointers are the right place for the current batch, head and sequence
    L += offset_batch_head_seq
    D += offset_batch_head_seq

    # array of indices which represent the offsets on dimension. We need to load all the dimensions.
    dim_offsets = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)

    # Index of the block in the sequence length to process
    start_kv = index_block_kv * BLOCK_SIZE_KV

    # array of indices which represent the offsets for the tokens in K and V sequence to process
    kv_offsets = start_kv + tl.arange(0, BLOCK_SIZE_KV)

    dV_block = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)

    # Load K and V in SRAM -> (BLOCK_SIZE_KV, HEAD_DIM)
    V_block = tl.load(
        V + kv_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )
    K_block = tl.load(
        K + kv_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )

    q_offsets = tl.arange(0, BLOCK_SIZE_Q)

    # We access Q as a transposed matrix -> (HEAD_DIM, BLOCK_SIZE_Q)
    # We point to the first BLOCK_SIZE_Q rows of Q for both q_T and dO pointers, inside the for loop we will move forward by BLOCK_SIZE_Q rows at each iteration
    Q_T_ptrs = Q + q_offsets[None, :] * stride_seq + dim_offsets[:, None] * stride_dim
    dO_ptrs = dO + q_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim

    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_SIZE_Q
    # Iterate over the SEQ_LEN of the Q matrix
    for _ in range(num_steps):

        # Load a block of Q transpose
        Q_T_block = tl.load(Q_T_ptrs)

        # load the logsumexp values for the queries in the current block
        q_offsets = curr_q + tl.arange(0, BLOCK_SIZE_Q)
        L_block = tl.load(L + q_offsets)

        # K(Q^T) = S^T
        S_T_block = softmax_factor * tl.dot(K_block, Q_T_block)
        # softmax with logsumexp trick, we need L^T
        P_T_block = tl.math.exp(S_T_block - L_block[None, :])

        if STAGE == 3:
            # For causal attention
            # Mask is TRUE for values that do not need to be masked -> (BLOCK_SIZE_KV, BLOCK_SIZE_Q)
            mask_block = q_offsets[None, :] >= kv_offsets[:, None]
            # Replace the masked values with 0
            # Not needed to mask with -inf before applying softmax as we already computed the normalization factors (stored in 'm')
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        # load dO block
        dO_block = tl.load(dO_ptrs)

        # Accumulate the dot product in dV_block
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # Di that we computed in the preprocessing step
        Di = tl.load(D + q_offsets)

        # We compute dP^T (not dP because we will need dS^T to update dK)
        dP_T_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # Element wise multiplication to get S^T
        dS_T_block = (dP_T_block - Di[None, :]) * P_T_block
        dS_T_block = dS_T_block.to(tl.float16)

        # We accumulate the dot product in dK_block
        dK_block = softmax_factor * tl.dot(dS_T_block, tl.trans(Q_T_block))

        # Note: tl.advance is usable only if the pointer was created using tl.make_ptr (not the case here)
        # Update the pointers
        curr_q += BLOCK_SIZE_Q
        Q_T_ptrs += BLOCK_SIZE_Q * stride_seq
        dO_ptrs += BLOCK_SIZE_Q * stride_seq

    # Store the computed dK and dV blocks
    dV_block_ptrs = (
        dV + kv_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )
    tl.store(dV_block_ptrs, dV_block)

    dK_block_ptrs = (
        dK + kv_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )
    tl.store(dK_block_ptrs, dK_block)


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_factor,
    dO,
    dQ,
    dK,
    dV,
    L,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Index representing the combination of batch and head to process
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (index_batch * stride_batch + index_head * stride_head).to(
        tl.int64
    )
    offset_batch_head_seq = (offset_batch_head * SEQ_LEN).to(tl.int64)

    # put the pointers are the right place for the current batch and head
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dQ += offset_batch_head
    dV += offset_batch_head
    dK += offset_batch_head
    dO += offset_batch_head_seq

    # put the pointers are the right place for the current batch, head and sequence
    L += offset_batch_head_seq
    D += offset_batch_head_seq

    # array of indices which represent the offsets on dimension. We need to load all the dimensions
    dim_offsets = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_q = index_block_kv * BLOCK_SIZE_Q
    q_offsets = start_q + tl.arange(0, BLOCK_SIZE_Q)

    Q_block = tl.load(
        Q + q_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )
    dQ_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + q_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )

    L_block = tl.load(L + q_offsets)
    L_block = L_block[:, None]

    kv_offsets = tl.arange(0, BLOCK_SIZE_KV)

    # We access the K and V as transposed blocks
    K_T_ptrs = K + kv_offsets[None, :] * stride_seq + dim_offsets[:, None] * stride_dim
    V_T_ptrs = V + kv_offsets[None, :] * stride_seq + dim_offsets[:, None] * stride_dim

    Di = tl.load(D + q_offsets)

    # We iterate over the K and V blocks
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_SIZE_KV

    for _ in range(num_steps):
        K_T_block = tl.load(K_T_ptrs)
        V_T_block = tl.load(V_T_ptrs)
        S_block = softmax_factor * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(S_block - L_block)

        if STAGE == 3:
            # For Causal Attention
            kv_offsets = curr_kv + tl.arange(0, BLOCK_SIZE_KV)
            mask_block = q_offsets[:, None] >= kv_offsets[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        # Compute dP and dS
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        # Accumulate the dot product in dQ_block
        dQ_block += softmax_factor * tl.dot(dS_block, tl.trans(K_T_block))

        # Update the pointers
        curr_kv += BLOCK_SIZE_KV
        K_T_ptrs += BLOCK_SIZE_KV * stride_seq
        V_T_ptrs += BLOCK_SIZE_KV * stride_seq

    dQ_block_ptrs = (
        dQ + q_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )
    # Single write to the HBM (compared to multiple writes in the loop in the paper)
    tl.store(dQ_block_ptrs, dQ_block)
