import torch


###################################### Multi-Head Attention ######################################


def multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode):
    """
    Implementation of MultiHead Attention which support:
    - Causal Attention
    - Global Attention
    - Sliding Window Attention
    """
    # Q, K, V are already: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    _, _, SEQ_LEN, HEAD_DIM = Q.shape

    softmax_factor = 1 / HEAD_DIM**0.5

    all_ones = torch.ones((SEQ_LEN, SEQ_LEN), device="cuda")

    if attn_mode == "causal":
        MASK = torch.tril(all_ones)

    elif attn_mode == "sliding_window":
        # window_right = (WINDOW_SIZE - 1) // 2
        # window_left = WINDOW_SIZE - 1 - window_right
        half_window = WINDOW_SIZE // 2
        MASK = torch.triu(all_ones, -1 * half_window) * torch.tril(
            all_ones, half_window
        )

    # Compute attention scores
    P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_factor

    # Add the computed mask
    if attn_mode == "causal" or attn_mode == "sliding_window":
        P[:, :, MASK == 0] = float("-inf")

    # Change P to float32 before softmax for numerical stability and precision and back to float16 afterwards
    attn_weights = torch.softmax(P.float(), dim=-1).half()
    attn_output = torch.matmul(attn_weights, V)

    return attn_output
