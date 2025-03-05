import torch


###################################### Multi-Head Attention ######################################


def multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode):
    """
    Implementation of MultiHead Self-Attention:
    - Global Attention
    - Causal Attention
    - Sliding Window Attention
    - No dropout
    - Float16
    """

    # Q, K, V are already: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    _, _, SEQ_LEN, HEAD_DIM = Q.shape

    attn_bias = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=Q.dtype, device=Q.device)
    softmax_factor = 1 / HEAD_DIM**0.5

    if attn_mode == "causal":
        MASK = torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=Q.device).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(MASK.logical_not(), float("-inf"))

    if attn_mode == "sliding_window":
        all_ones = torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=Q.device)
        half_window = WINDOW_SIZE // 2
        MASK = torch.triu(all_ones, diagonal=-half_window) & torch.tril(
            all_ones, diagonal=half_window
        )
        attn_bias.masked_fill_(MASK.logical_not(), float("-inf"))

    P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_factor
    P += attn_bias
    # Change P to float32 before softmax for numerical stability and precision and back to float16 afterwards
    attn_weights = torch.softmax(P.float(), dim=-1).half()
    attn_output = torch.matmul(attn_weights, V)

    return attn_output
