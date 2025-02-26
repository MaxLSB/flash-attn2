import torch

###################################### Multi-Head Attention ######################################


class MultiHeadAttention:

    @staticmethod
    def forward(Q, K, V, causal):
        # Q, K, V are already: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        _, _, SEQ_LEN, HEAD_DIM = Q.shape

        softmax_factor = 1 / HEAD_DIM**0.5

        P = (
            torch.matmul(Q, K.transpose(-2, -1)) * softmax_factor
        )  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)

        # It's causal so the upper triangle of the matrix is filled with infinite values
        if causal:
            MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN)))  # (seq_len, seq_len)
            P[:, :, MASK == 0] = float("-inf")

        # Change P to float32 before softmax for numerical stability and precision and back to float16 afterwards
        attn_weights = torch.softmax(P.float(), dim=-1).half()

        attn_output = torch.matmul(P, V)

        return attn_output, attn_weights
