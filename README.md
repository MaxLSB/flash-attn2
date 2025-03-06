# FlashAttention-2 in Triton with Sliding Window Attention

## Introduction

This repository provides an implementation of FlashAttention-2 **Forward and Backward Pass** for self-attention in Triton and which handles **Sliding Window Attention**. FlashAttention-2 is a memory-efficient algorithm for computing attention that significantly reduces memory bandwidth requirements and improves performance on GPU hardware.

This implementation supports several configurations for both Forward and Backward Pass including:

- **Sliding Window Attention**
- **Global Attention**
- **Causal Attention**

Note:
- Partially inspired by OpenAI's Fused Attention in Triton.
- This implementation is intended for educational purposes and can be optimized.
- No dropout is applied.
- Uses FP16 precision.

## Benchmarking

<div align="center">
  <img src="media/benchmark.png" alt="FlashAttention-2 Benchmarks" width="500" />
</div>

<br>

This implementation replicates the trend described in the paper with significant performance improvements compared to traditional attention mechanisms.

- Up to 2-10x speedup compared to a standard PyTorch attention implementation

## Implementation Details

The implementation leverages Triton's capabilities to generate efficient CUDA kernels on the fly. Key optimizations include:

- Block-sparse computation patterns for sliding window attention
- Memory-efficient backward pass with gradient accumulation
- Fused softmax operations to reduce memory bandwidth
- Optimized tiling strategies for different attention patterns

## Testing

Test file to ensure that the results from PyTorch and Triton implementation match:

```bash
python -m tests.test --attn_mode 'sliding_window' --window_size 1000
python -m tests.test --attn_mode 'causal'
python -m tests.test --attn_mode 'global'
```

Each test verifies:
- Numerical accuracy against a Standard PyTorch implementation
- Gradient correctness

## Installation

```bash
git clone https://github.com/MaxLSB/flash-attention-2.git
```

# To Do

- Fix the current restrictions for Sliding Window:
$$ \text{SEQ\_LEN} \geq 4 \times \text{BLOCK\_SIZE} $$
$$ 2 \times \text{BLOCK\_SIZE} \leq \text{WINDOW\_SIZE} \leq \text{SEQ\_LEN} $$
- Improve the Autotune in the backward pass




<!-- Other ideas:
- Multi-Head Latent Flash Attention
- GQA Flash Attention
- Native Sparse Attention with Flash Attention -->