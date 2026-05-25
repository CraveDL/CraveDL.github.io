---
layout: post
title: "FlashAttention Explained: IO-Aware Attention, Online Softmax, and MHA/MQA/GQA"
date: 2026-05-25
categories: [LLM, Transformer, Attention, GPU]
---

Attention is the core operation inside Transformer models. It is also one of the first bottlenecks you meet when sequence length grows.

At a high level, attention looks simple:

```text
S = QK^T
P = softmax(S)
O = PV
```

But a straightforward implementation materializes large intermediate matrices. FlashAttention makes the same attention computation faster and more memory-efficient by changing how the computation is scheduled on GPU memory.

This article explains why standard attention is memory-heavy, how FlashAttention avoids writing the full attention matrix to HBM, how online softmax makes tiled attention exact, and how MHA, MQA, and GQA relate to attention memory.

<figure>
  <img src="/assets/images/flash-attention/attention-pipeline.svg" alt="Scaled dot-product attention pipeline">
  <figcaption>Figure 1. Standard attention computes scores, applies row-wise softmax, then mixes value vectors.</figcaption>
</figure>

## The Standard Attention Computation

For one attention head, assume:

```text
Q, K, V in R^(N x d)
```

where `N` is sequence length and `d` is head dimension.

The attention output is:

```text
S = QK^T
P = softmax(S)
O = PV
```

The score matrix `S` has shape:

```text
N x N
```

The probability matrix `P` also has shape:

```text
N x N
```

This is the main problem. As sequence length grows, the intermediate attention matrices grow quadratically.

For example, if `N = 8192`, then `S` contains:

```text
8192 x 8192 = 67,108,864 elements
```

That is for one head and one batch item before considering dtype, dropout masks, gradients, and multiple layers.

## Compute-Bound vs Memory-Bound Operations

Not every GPU operation is bottlenecked by the same resource.

Compute-bound operations are limited mainly by arithmetic throughput. Large matrix multiplications and multi-channel convolutions often fall into this category.

Memory-bound operations are limited mainly by data movement. Elementwise operations, reductions, dropout, masking, and softmax often spend much of their time reading from and writing to memory.

Attention contains both:

- `QK^T` and `PV` are matrix multiplications.
- masking, softmax, dropout, and intermediate matrix writes are memory-heavy.

The core FlashAttention insight is that attention performance is not only about FLOPs. It is also about IO: how often tensors move between high-bandwidth memory and fast on-chip memory.

## The Memory Problem in Naive Attention

Modern GPUs have different memory levels.

HBM is large but relatively slow compared with on-chip SRAM. SRAM is much smaller, but it is much faster.

A naive attention implementation may do something like:

1. Load `Q` and `K` from HBM to SRAM.
2. Compute `S = QK^T`.
3. Write `S` back to HBM.
4. Load `S` back to SRAM.
5. Compute `P = softmax(S)`.
6. Write `P` back to HBM.
7. Load `P` and `V` from HBM to SRAM.
8. Compute `O = PV`.
9. Write `O` back to HBM.

The expensive part is that `S` and `P` are full `N x N` matrices.

<figure>
  <img src="/assets/images/flash-attention/naive-attention-memory.svg" alt="Naive attention writes large intermediate matrices through HBM">
  <figcaption>Figure 2. Naive attention repeatedly moves large intermediate matrices through HBM.</figcaption>
</figure>

For long sequences, caching full attention scores is costly. The memory traffic grows quickly, even when the arithmetic is not the only bottleneck.

## Kernel Fusion Helps, but It Is Not Enough

For memory-bound workloads, a common optimization is fusion.

Instead of launching separate kernels for mask, softmax, dropout, and other operations, a fused kernel combines multiple steps and avoids writing unnecessary intermediate results.

Fusion improves efficiency, but training often still needs intermediate values for backward propagation. A basic fused implementation may reduce kernel overhead, but it does not fully solve the `N x N` attention matrix problem.

FlashAttention goes further by redesigning the attention schedule around memory hierarchy.

## FlashAttention: IO-Aware Exact Attention

FlashAttention computes exact attention. It is not an approximation.

The key idea is tiling:

- Split `Q` into row blocks.
- Split `K` and `V` into column blocks.
- Load one block of `K` and `V` into SRAM.
- Stream blocks of `Q` through SRAM.
- Compute local score blocks.
- Update the output block incrementally.
- Never materialize full `S` or full `P` in HBM.

<figure>
  <img src="/assets/images/flash-attention/flashattention-tiling.svg" alt="FlashAttention tile-based computation over Q K and V blocks">
  <figcaption>Figure 3. FlashAttention computes attention block by block inside SRAM and updates output incrementally.</figcaption>
</figure>

The result is mathematically equivalent to standard attention, but with far less HBM traffic.

This matters because HBM reads and writes are often the limiting factor for attention.

## The Tiling Algorithm

FlashAttention chooses block sizes based on available on-chip SRAM.

The algorithm keeps the following data:

- output blocks `O_i`
- row-wise max values `m_i`
- row-wise normalization sums `l_i`
- blocks of `Q_i`, `K_j`, and `V_j`

For each `K_j, V_j` block, it iterates over `Q_i` blocks:

1. Load `K_j` and `V_j` from HBM to SRAM.
2. Load `Q_i`, `O_i`, `m_i`, and `l_i` from HBM to SRAM.
3. Compute a local score block: `S_ij = Q_i K_j^T`.
4. Compute local row maxima and exponentiated scores.
5. Update the running softmax statistics.
6. Update the output block `O_i`.
7. Write the updated `O_i`, `m_i`, and `l_i` back to HBM.

Because each block fits in SRAM, the large intermediate matrix does not need to be stored in HBM.

## Why Online Softmax Is Needed

Softmax is normally computed across an entire row:

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

If the row is split into blocks, you cannot simply softmax each block independently. The denominator must include all blocks in the row.

FlashAttention solves this with online softmax.

For each row, it maintains:

- `m`: the running maximum score
- `l`: the running sum of exponentials after max correction
- `O`: the running output

When a new block arrives, the algorithm updates the maximum and rescales the previous sum and output so the final result matches full softmax.

<figure>
  <img src="/assets/images/flash-attention/online-softmax.svg" alt="Online softmax statistics for tiled attention">
  <figcaption>Figure 4. Online softmax lets tiled blocks combine into the same result as full-row softmax.</figcaption>
</figure>

This is the trick that makes FlashAttention exact instead of approximate.

## Why It Reduces Memory

Naive attention stores or reloads:

- `S = QK^T`
- `P = softmax(S)`

Both are `N x N`.

FlashAttention stores:

- the final output `O`
- running row maxima `m`
- running row sums `l`
- small blocks currently being processed

This changes the memory behavior dramatically. The output remains `N x d`, while the large `N x N` intermediate matrices are avoided.

The arithmetic complexity is still attention-like, but the IO complexity is much better.

## Multi-Head Attention

Multi-head attention, or MHA, uses multiple attention heads. Each head has its own projections for queries, keys, and values.

This gives the model more expressive capacity. Different heads can learn different attention patterns.

The tradeoff is memory. During decoding, each layer stores a KV cache for every key/value head.

If the model has many heads and a long context, KV cache memory becomes a major inference bottleneck.

## Multi-Query Attention

Multi-query attention, or MQA, keeps multiple query heads but shares a single key/value head across them.

This greatly reduces KV cache size.

The tradeoff is that sharing all K/V heads can reduce modeling capacity compared with full MHA.

MQA is useful when inference speed and memory are critical.

## Grouped-Query Attention

Grouped-query attention, or GQA, sits between MHA and MQA.

Instead of one K/V head per query head, and instead of one shared K/V head for all query heads, GQA groups query heads. Each group shares one K/V head.

<figure>
  <img src="/assets/images/flash-attention/mha-mqa-gqa.svg" alt="Comparison of MHA MQA and GQA head sharing">
  <figcaption>Figure 5. MHA maximizes per-head K/V capacity; MQA minimizes KV cache; GQA balances the two.</figcaption>
</figure>

GQA is a practical compromise:

- less KV cache than MHA
- more capacity than MQA
- good fit for long-context LLM inference

FlashAttention and MQA/GQA solve related but different problems. FlashAttention improves how attention is computed. MQA and GQA reduce how much K/V state the model needs to store and read.

## FlashAttention and Training

During training, memory pressure is not only from activations in the forward pass. Backpropagation also needs intermediate values.

Naive attention may store the attention matrix for backward computation. FlashAttention avoids storing full attention probabilities and recomputes needed blocks during the backward pass.

This is another memory-throughput tradeoff:

- store fewer intermediate tensors
- move less data through HBM
- recompute small blocks when needed

The result is usually faster and more memory efficient for long sequences.

## Practical Mental Model

You can think of FlashAttention as three ideas working together:

1. Tiling: make the working set fit in fast on-chip SRAM.
2. Online softmax: compute exact softmax without seeing the full row at once.
3. Recomputation: avoid saving large intermediate tensors for backward.

The core target is IO efficiency. Instead of trying to reduce the mathematical definition of attention, FlashAttention keeps the math and reduces unnecessary memory movement.

## Comparing Attention Implementations

| Method | Stores Full Attention Matrix | Main Optimization | Typical Benefit |
| --- | --- | --- | --- |
| Naive attention | yes | straightforward implementation | simple but memory-heavy |
| Fused attention kernels | often reduced | combine operations into fewer kernels | less kernel overhead and less intermediate traffic |
| FlashAttention | no | tiling plus online softmax | much lower HBM traffic and better long-sequence efficiency |
| MQA | no direct change to kernel | share K/V across query heads | smaller KV cache |
| GQA | no direct change to kernel | group query heads per K/V head | KV cache reduction with better capacity than MQA |

## Key Takeaways

FlashAttention is not a new attention formula. It is a better way to compute the same attention operation on GPU hardware.

The main lessons are:

- Standard attention creates `N x N` score and probability matrices.
- Those matrices create heavy HBM traffic as sequence length grows.
- GPU memory hierarchy matters: SRAM is fast but small; HBM is large but slower.
- FlashAttention tiles `Q`, `K`, and `V` so each working block fits on-chip.
- Online softmax makes block-wise attention mathematically equivalent to full softmax.
- The algorithm avoids materializing full `S` and `P` matrices in HBM.
- MHA, MQA, and GQA are complementary design choices that affect KV cache memory.
- FlashAttention optimizes attention computation; MQA/GQA optimize attention state size.

For long-context LLMs, attention efficiency is not only about FLOPs. It is about moving less data, keeping the right blocks close to the compute units, and avoiding quadratic intermediate memory whenever possible.
