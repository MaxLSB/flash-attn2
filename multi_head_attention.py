import torch

###################################### Multi-Head Attention ######################################


def multi_head_attention(Q, K, V, causal):

    # Q, K, V are already: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    _, _, SEQ_LEN, HEAD_DIM = Q.shape

    softmax_factor = 1 / HEAD_DIM**0.5

    MASK = torch.tril(
        torch.ones((SEQ_LEN, SEQ_LEN), device="cuda")
    )  # (seq_len, seq_len)

    P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_factor

    # It's causal so the upper triangle of the matrix is filled with infinite values
    if causal:
        P[:, :, MASK == 0] = float("-inf")

    # Change P to float32 before softmax for numerical stability and precision and back to float16 afterwards
    attn_weights = torch.softmax(P.float(), dim=-1).half()
    attn_output = torch.matmul(attn_weights, V)

    return attn_output
