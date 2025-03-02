import numpy as np
import triton

WINDOW_SIZE = 11
SEQ_LEN = 12
BLOCK_SIZE_Q = 2

# [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
# [1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
# [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
# [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
# [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
# [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
# [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
# [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
# [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
# [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
# [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
# [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]

# [2, 2, 2, 1, 0, 0],
# [2, 2, 2, 2, 1, 0],
# [2, 2, 2, 2, 2, 1],
# [1, 2, 2, 2, 2, 2],
# [0, 1, 2, 2, 2, 2];
# [0, 0, 1, 2, 2, 2];


NUM_BLOCKS_Q = int(np.ceil(SEQ_LEN // BLOCK_SIZE_Q))
window_block_index = int(np.ceil((WINDOW_SIZE - BLOCK_SIZE_Q) / (2 * BLOCK_SIZE_Q)))
window_block_index = triton.cdiv(WINDOW_SIZE - BLOCK_SIZE_Q, 2 * BLOCK_SIZE_Q)


def stage_3(block_index_q):

    if block_index_q - window_block_index >= 0:
        lower = (block_index_q - window_block_index) * BLOCK_SIZE_Q
        higher = lower + BLOCK_SIZE_Q
    else:
        lower, higher = 0, 0

    print(f"> block_index_q={block_index_q}:\nlower={lower}, higher={higher}\n")


def stage_2(block_index_q):

    if block_index_q + window_block_index <= NUM_BLOCKS_Q - 1:
        lower = (block_index_q + window_block_index) * BLOCK_SIZE_Q
        higher = lower + BLOCK_SIZE_Q
    else:
        lower, higher = 0, 0

    print(f"> block_index_q={block_index_q}:\nlower={lower}, higher={higher}\n")


# STAGE 1:
def stage_1(block_index_q):
    lower = max(0, block_index_q - window_block_index + 1) * BLOCK_SIZE_Q
    higher = (min(NUM_BLOCKS_Q, block_index_q + window_block_index)) * BLOCK_SIZE_Q
    print(f"> block_index_q={block_index_q}:\nlower={lower}, higher={higher}\n")


# stage_3(block_index_q=0)
# stage_3(block_index_q=1)
# stage_3(block_index_q=2)
# stage_3(block_index_q=3)
# stage_3(block_index_q=4)
# stage_3(block_index_q=5)

##########################################################################

WINDOW_SIZE = 11
half_window = WINDOW_SIZE // 2
q_offsets = np.array([0, 1])
start_kv = 6
kv_offsets = np.array([0, 1])
print(half_window)

mask_2 = q_offsets[:, None] + half_window >= (start_kv + kv_offsets[None, :])
mask_3 = q_offsets[:, None] - half_window <= (start_kv + kv_offsets[None, :])

print("mask_2:")
print(mask_2)
# print("mask_3:")
# print(mask_3)
