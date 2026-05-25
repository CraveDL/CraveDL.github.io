---
layout: post
title: "Positional Encoding in Transformers: Absolute, Relative, RoPE, ALiBi, and Long-Context Extrapolation"
date: 2026-05-25
categories: [Transformer, LLM, Positional-Encoding]
---

Transformers process a sequence as a set of token representations. Without positional information, self-attention has no built-in sense of token order. A sentence can start to look like a bag of words.

Positional encoding solves that problem. It injects order into the model so the Transformer can distinguish "the model saw the token at position 3" from "the model saw the same token at position 300."

This article explains the major positional encoding families used in Transformer models: absolute positional encoding, relative positional encoding, RoPE, ALiBi, and long-context extrapolation techniques such as Position Interpolation and NTK-aware scaled RoPE.

<figure>
  <img src="/assets/images/positional-encoding/embedding-plus-position.svg" alt="Token embedding plus positional encoding before the Transformer">
  <figcaption>Figure 1. Token embeddings represent meaning; positional encodings inject order.</figcaption>
</figure>

## Why Positional Encoding Is Needed

Classic neural architectures encode order through structure.

RNNs process tokens recursively, so the hidden state naturally depends on previous positions. CNNs use convolution windows and pooling, so local order is partially encoded by the computation pattern.

Transformers are different. Self-attention compares every token with every other token, but the attention operation itself does not know whether one token came before or after another unless position is represented somewhere.

That creates the central design question:

> How should a Transformer know where a token is?

There are two broad answers:

- Add positional information to the input representation. This is the usual absolute positional encoding approach.
- Modify the attention computation so the model can reason about relative distance. This is the relative positional encoding approach.

<figure>
  <img src="/assets/images/positional-encoding/absolute-vs-relative.svg" alt="Absolute positional encoding compared with relative positional encoding">
  <figcaption>Figure 2. Absolute encodings label positions directly; relative encodings affect pairwise attention.</figcaption>
</figure>

## Absolute Positional Encoding

Absolute positional encoding assigns each position a vector. The vector is added to the token embedding before the sequence enters the Transformer.

In simplified form:

```text
input_vector(position i) = token_embedding(i) + position_embedding(i)
```

Absolute positional encoding is easy to understand and simple to implement. The main variants are:

- sinusoidal positional encoding
- learned positional encoding
- recurrent or continuous positional encoding

## Sinusoidal Positional Encoding

The original Transformer paper introduced sinusoidal positional encoding. It uses sine and cosine functions at different frequencies.

A simplified version of the formula is:

```text
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
```

where `pos` is the token position, `i` is the dimension index, and `d_model` is the hidden size.

<figure>
  <img src="/assets/images/positional-encoding/sinusoidal-waves.svg" alt="Sinusoidal positional encoding waves at different frequencies">
  <figcaption>Figure 3. Sinusoidal encoding uses many frequencies to give each position a structured signature.</figcaption>
</figure>

Sinusoidal encoding has two useful properties:

- It is deterministic, so it does not require learned position parameters.
- It can compute encodings for positions longer than those seen during training.

Because the encoding is built from periodic functions, it has some theoretical extrapolation ability. In practice, however, extrapolation quality still depends on how the model learned to use those signals.

## Learned Absolute Positional Encoding

Learned positional encoding is even simpler. If the maximum sequence length is 512 and the hidden size is 768, the model learns a position embedding matrix of shape:

```text
512 x 768
```

Each position has a trainable vector. This vector is added to the token embedding.

This method was widely used in early Transformer models such as BERT, GPT-style models, and ALBERT.

The weakness is length extrapolation. If the matrix was trained for 512 positions, the model has no naturally learned vector for position 2048. Extending the matrix after pretraining can break the positional structure the model learned.

## Recurrent and Continuous Positional Encoding

RNNs do not need separate positional encodings in the same way because recurrence already carries order. This inspired another idea: use a recurrent or continuous model to generate position representations and then feed them into the Transformer.

One line of work models positional encoding with continuous dynamical systems. For example, FLOATER uses an ODE-based approach to encode positions and relationships between positions.

The benefit is better extrapolation behavior. The cost is that recurrence or continuous dynamics may reduce parallelism and introduce speed bottlenecks.

## Relative Positional Encoding

Relative positional encoding does not focus on the absolute position of a token. It focuses on the relationship between two tokens.

This is often more natural for language. In many linguistic patterns, relative order matters more than absolute index. A word being three positions away may be more meaningful than it being at global position 128.

Relative positional encoding changes the attention computation so attention scores can depend on distance.

## Classic Relative Position Representations

The classic formulation comes from Google's work on self-attention with relative position representations. The model adds relative position information into the attention calculation so token `i` can attend to token `j` differently depending on `j - i`.

This style of encoding was also used in models such as NEZHA.

The main advantage is that attention becomes distance-aware. The model can learn patterns such as:

- nearby words often matter more
- certain syntactic dependencies happen at short ranges
- long-range dependencies may need different treatment

## Transformer-XL and XLNet-Style Relative Position Encoding

Transformer-XL introduced a relative positional encoding mechanism designed for language modeling beyond a fixed-length context. XLNet inherited related ideas.

The design separates content-based attention and position-based attention, allowing the model to represent long-distance dependencies more flexibly.

Its advantages include:

- stronger long-distance dependency modeling
- better handling of variable sequence lengths
- no hard truncation of relative positions when sinusoidal generation is used

This makes it more suitable for long text tasks than simple fixed absolute embeddings.

## T5-Style Relative Position Bias

T5 uses a simpler relative position approach. Instead of adding full relative vectors into every attention interaction, it uses learned relative attention bias values.

The key idea is bucketing. Relative distances are mapped into a smaller number of buckets. Nearby distances can be represented precisely, while far distances are grouped more coarsely.

This reduces computation and parameter complexity.

T5-style relative bias has three practical strengths:

- It is simple.
- It scales well to longer text.
- It lets the model learn how much attention should shift based on distance.

## DeBERTa's Disentangled Attention

DeBERTa takes a different path. Its attention mechanism separates content and position representations.

T5 keeps a simplified position bias. DeBERTa keeps richer content-position and position-content interactions, while removing the pure position-position term.

This lets the model represent:

- content-to-content attention
- content-to-position interaction
- position-to-content interaction

The result is a more expressive attention mechanism, especially for tasks where token identity and relative position interact strongly.

## RoPE: Rotary Position Embedding

RoPE, or Rotary Position Embedding, is one of the most important positional encoding methods in modern LLMs.

RoPE injects position by rotating query and key vectors. For each token position, the model applies a position-dependent rotation to the query and key representations before computing the attention dot product.

<figure>
  <img src="/assets/images/positional-encoding/rope-rotation.svg" alt="RoPE rotates query and key vectors according to position">
  <figcaption>Figure 4. RoPE rotates query and key vectors; the dot product naturally depends on relative distance.</figcaption>
</figure>

In a two-dimensional subspace, a rotation looks like:

```text
[x1'] = [cos theta  -sin theta] [x1]
[x2']   [sin theta   cos theta] [x2]
```

For high-dimensional vectors, RoPE splits the vector into pairs of dimensions and applies a small rotation matrix to each pair.

This is efficient because the high-dimensional rotation matrix is sparse and structured. Implementations do not need to materialize the full matrix.

RoPE has several practical advantages:

- It is efficient.
- It is easy to implement.
- It gives attention a relative-position property.
- It works naturally with decoder-only LLMs.
- It can be used with linear attention variants.
- It has a form of long-distance decay.

Many modern LLMs use RoPE or RoPE variants.

## ALiBi: Attention with Linear Biases

ALiBi stands for Attention with Linear Biases. It was proposed in "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation."

Unlike standard absolute encodings, ALiBi does not add a position vector at the embedding layer. Instead, it adds a static, non-learned linear bias to attention scores.

The bias penalizes attention based on distance. Tokens farther away receive a larger negative bias.

This encourages the model to learn local attention patterns while still allowing long-context extrapolation.

ALiBi is attractive because it requires little structural change. However, direct length extrapolation can still fail in practice. When the test context is much longer than training context, perplexity can increase sharply unless the model is adapted carefully.

## Long-Context Extrapolation

Long-context extrapolation asks:

> Can a model trained on a limited context length handle a much longer context during inference?

There are two related problems:

- The model may see positional values it never saw during training.
- The attention mechanism must process far more tokens than it handled during training.

RoPE can mathematically compute encodings for arbitrary positions, but that does not guarantee stable model behavior. In practice, when test length exceeds training length, metrics such as perplexity can degrade significantly.

<figure>
  <img src="/assets/images/positional-encoding/context-extension.svg" alt="Direct extrapolation, position interpolation, and NTK-aware scaled RoPE">
  <figcaption>Figure 5. Long-context extension methods try to avoid extrapolation collapse while preserving local resolution.</figcaption>
</figure>

## Position Interpolation

Position Interpolation, or PI, was proposed to extend the context window of large models.

The core idea is to scale down position indices so a longer sequence is mapped into the positional range the model saw during pretraining.

For example, if a model was trained with context length 2048 and you want to use 8192 tokens, you can map the new positions into the original range:

```text
scaled_position = original_position / scale_factor
```

This avoids directly extrapolating to far unseen positions.

The tradeoff is local resolution. Compressing positions means nearby tokens become closer in positional space. Language modeling depends heavily on local order, so too much compression can damage the model's ability to distinguish neighboring tokens.

Empirically, PI can be effective with a small amount of long-context fine-tuning. It is much more efficient than trying to extend context length by direct fine-tuning alone.

## NTK-Aware Scaled RoPE

Position Interpolation is linear. NTK-aware scaled RoPE uses a more nuanced approach.

Instead of directly scaling every position by the same factor, it changes the RoPE base. This changes the rotation speed across RoPE dimensions.

The intuition is:

- high-frequency dimensions are important for local token resolution
- low-frequency dimensions are useful for long-range position coverage

So the method can be summarized as:

```text
high frequency: extrapolate
low frequency: interpolate
```

This helps preserve local discrimination while extending the usable context window.

NTK-aware scaling became popular because it can extend LLaMA-style RoPE context lengths with little or no fine-tuning, while keeping perplexity degradation smaller than direct extrapolation.

## Comparing Positional Encoding Methods

| Method | Where Position Enters | Main Strength | Main Weakness |
| --- | --- | --- | --- |
| Learned absolute encoding | input embeddings | simple and trainable | poor length extrapolation |
| Sinusoidal encoding | input embeddings | deterministic and extrapolatable in form | extrapolation is not always stable in trained models |
| Classic relative encoding | attention calculation | distance-aware attention | more complex attention computation |
| T5 relative bias | attention logits | simple and efficient | less expressive than full relative interaction |
| DeBERTa disentangled attention | attention calculation | rich content-position interaction | more complex architecture |
| RoPE | query/key rotation | efficient relative-position behavior | long-context extrapolation still needs care |
| ALiBi | attention bias | simple long-context bias | may need adaptation for very long extrapolation |
| PI | position rescaling | efficient context extension | compresses local resolution |
| NTK-aware scaled RoPE | RoPE frequency scaling | balances local and long-range behavior | more subtle to configure |

## Practical Guidelines

For most LLM engineering work, you should not change positional encoding casually. It is deeply tied to model pretraining.

Use these guidelines:

- If you train a small Transformer from scratch, learned absolute positions are simple, but watch the maximum length.
- If you need deterministic position vectors, sinusoidal encoding remains a useful baseline.
- If the task depends heavily on relative order, consider relative bias or relative position attention.
- For decoder-only LLMs, RoPE is often the default modern choice.
- For long-context extension of RoPE-based models, evaluate PI and NTK-aware scaling before direct extrapolation.
- Always test perplexity and downstream task quality at the target context length.

The most important evaluation is not whether the formula can produce longer positions. It is whether the trained model can use those positions reliably.

## Key Takeaways

Positional encoding is what gives a Transformer an ordered view of a sequence.

The main lessons are:

- Self-attention needs positional signals to avoid becoming order-blind.
- Absolute encodings add position vectors to token embeddings.
- Relative encodings modify attention based on token distance.
- Learned absolute embeddings are simple but weak at extrapolation.
- Sinusoidal encodings can generate arbitrary positions, but learned behavior may still fail beyond training length.
- RoPE encodes position by rotating query and key vectors, producing relative-position behavior efficiently.
- ALiBi adds distance-based linear bias directly to attention scores.
- Long-context extrapolation is constrained by both positional encoding and attention behavior.
- PI compresses long positions into the trained range.
- NTK-aware scaled RoPE preserves local resolution while extending long-range capacity.

For long-context LLMs, positional encoding is not just a formula. It is part of the model's memory, attention geometry, and inference stability.
