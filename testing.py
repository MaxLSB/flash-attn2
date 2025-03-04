import numpy as np
import triton

WINDOW_SIZE = 40
SEQ_LEN = 128
BLOCK_SIZE_Q = 16

half_window = WINDOW_SIZE // 2
window_block_right = triton.cdiv(1 + half_window, BLOCK_SIZE_Q)
window_block_left = triton.cdiv(half_window, BLOCK_SIZE_Q)
NUM_BLOCKS_Q = triton.cdiv(SEQ_LEN, BLOCK_SIZE_Q)

print(f"window_block_right={window_block_right}")
print(f"window_block_left={window_block_left}")

# LEFT DIAGONAL
# issue here most probably
def stage_3(block_index_q):
    
    if (block_index_q + 1) * BLOCK_SIZE_Q - 1 - half_window > 0:
        higher = max(0, block_index_q - window_block_left + 1) * BLOCK_SIZE_Q
    else:
        higher = 0
    
    if higher > 0:
        lower = higher - BLOCK_SIZE_Q
    else:
        lower = higher

    print(f"lower={lower}, higher={higher}")

1, 1, 1, 0, 0, 0
1, 1, 1, 1, 0, 0
1, 1, 1, 1, 1, 0
0, 1, 1, 1, 1, 1
0, 0, 1, 1, 1, 1
0, 0, 0, 1, 1, 1

# RIGHT DIAGONAL
def stage_2(block_index_q):

    if (half_window + 1) % BLOCK_SIZE_Q == 0:
        lower = min(NUM_BLOCKS_Q, block_index_q + window_block_right) * BLOCK_SIZE_Q
    else:
        lower = min(NUM_BLOCKS_Q, block_index_q + window_block_right - 1) * BLOCK_SIZE_Q
    
    if lower < NUM_BLOCKS_Q * BLOCK_SIZE_Q:
        higher = lower + BLOCK_SIZE_Q
    else:
        higher = lower

    print(f"lower={lower}, higher={higher}")


# IN BETWEEN
def stage_1(block_index_q):
    if (block_index_q + 1) * BLOCK_SIZE_Q - 1 - half_window > 0:
        lower = max(0, block_index_q - window_block_left + 1) * BLOCK_SIZE_Q
    else:
        lower = 0
    
    if (half_window + 1) % BLOCK_SIZE_Q == 0:
        higher = min(NUM_BLOCKS_Q, block_index_q + window_block_right) * BLOCK_SIZE_Q
    else:
        higher = min(NUM_BLOCKS_Q, block_index_q + window_block_right - 1) * BLOCK_SIZE_Q
    print(f"lower={lower}, higher={higher}")


for i in range(NUM_BLOCKS_Q):
    print(f'block_index_q={i}:')
    stage_3(block_index_q=i)
    stage_1(block_index_q=i)
    stage_2(block_index_q=i)
    

