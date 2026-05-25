---
layout: post
title: "Understanding Tokenizers for LLMs: From Words and Characters to BPE, WordPiece, and Unigram"
date: 2026-05-25
categories: [LLM, NLP, Tokenization]
---

Before a Transformer can process text, the text has to become numbers. That conversion is the job of a tokenizer.

Tokenization looks like a preprocessing detail, but it shapes almost everything downstream: vocabulary size, sequence length, memory cost, multilingual coverage, out-of-vocabulary behavior, and even how well a model learns rare words or domain-specific terms.

This article explains what tokenizers do, why subword tokenization became dominant in modern language models, and how common algorithms such as BPE, byte-level BPE, WordPiece, SentencePiece, and Unigram work.

## Transformer Input: Text Becomes Token IDs

A Transformer does not directly understand raw text. It receives a sequence of token IDs.

The high-level pipeline is:

```text
raw text -> tokenizer -> token IDs -> embedding layer -> Transformer
```

For example, a sentence such as:

```text
This is an input.
```

may become a token sequence:

```text
[CLS], This, is, an, input, ., [SEP]
```

and then a numeric ID sequence:

```text
101, 2023, 2003, 2019, 7953, 1012, 102
```

The embedding layer maps each token ID to a dense vector. That embedding layer is part of the Transformer model, but the tokenizer defines the vocabulary that the embedding layer must support.

This is why tokenizer design matters. A bad tokenizer can make sequences unnecessarily long, lose important information, or create too many unknown tokens.

## What a Tokenizer Does

A tokenizer has two responsibilities:

1. Split text into units called tokens.
2. Map each token to an integer ID in a vocabulary.

In LLM training and fine-tuning, tokenization is not optional. Every training sample must be converted into token IDs before it can enter the model.

The difficult part is deciding what a token should be.

Should a token be a word? A character? A byte? A frequent subword unit? The answer changes the tradeoff between vocabulary size and sequence length.

## Three Tokenization Granularities

There are three common tokenization granularities:

- word-based tokenization
- character-based tokenization
- subword-based tokenization

Each one makes a different compromise.

## Word-Based Tokenizers

A word-based tokenizer splits text into words and punctuation.

For a simple sentence, this feels natural:

```text
I love tokenizers!
```

could become:

```text
I, love, tokenizers, !
```

This matches human intuition, but it has serious engineering problems.

### Advantages

Word-based tokenization is easy to understand. Tokens are usually meaningful units, and each token often carries more information than a single character.

### Problems

The first problem is vocabulary size. Natural languages contain many words, inflections, names, product terms, spelling variants, and domain-specific expressions. If every word gets its own vocabulary entry, the vocabulary becomes very large.

The second problem is unknown words. If the vocabulary is capped at 10,000 or 50,000 words, rare words must be replaced by a special token such as `[UNK]`. That causes information loss.

The third problem is rule complexity. Consider:

```text
can't
```

Should it be one token, two tokens, or three?

```text
can't
can, n't
ca, n't
```

Once contractions, punctuation, abbreviations, code, names, and multilingual text are included, rule-based word tokenization becomes messy.

## Character-Based Tokenizers

A character-based tokenizer splits text into individual characters.

For example:

```text
What is your name?
```

could become:

```text
W, h, a, t,  , i, s,  , y, o, u, r,  , n, a, m, e, ?
```

### Advantages

Character-based tokenization can represent almost any text without unknown words, as long as the character set is covered.

For English, the vocabulary can be small. A basic English character vocabulary may need fewer than 256 symbols.

### Problems

Character tokens carry less semantic information than words or subwords. The model has to learn longer-range combinations just to reconstruct word-level meaning.

Character tokenization also produces much longer sequences. Longer sequences increase training cost, inference cost, and attention memory usage.

For Chinese, Japanese, emoji, and full Unicode coverage, the character vocabulary can also become large if the tokenizer operates directly at the Unicode character level.

## Why Subword Tokenization Became Dominant

Subword tokenization sits between word-level and character-level tokenization.

The core idea is simple:

- frequent words can remain whole tokens
- rare words can be split into smaller meaningful or reusable units
- unknown words can often be represented through subword pieces

For example:

```text
tokenization
```

may become:

```text
token, ization
```

or:

```text
token, ##ization
```

depending on the tokenizer.

Subword tokenization gives models a better tradeoff:

- vocabulary is smaller than word-level tokenization
- sequences are shorter than character-level tokenization
- rare words can be decomposed instead of replaced by `[UNK]`
- morphology and shared word parts can be reused

This is why modern language models typically rely on subword tokenizers.

## Common Subword Tokenizers

Four important subword tokenization families are:

- Byte-Pair Encoding
- Byte-level Byte-Pair Encoding
- WordPiece
- Unigram

SentencePiece is also important, but it is best understood as a tokenizer framework that can train algorithms such as BPE and Unigram directly from raw text.

## Byte-Pair Encoding

Byte-Pair Encoding, or BPE, was originally a compression algorithm. In NLP, it is used to build a subword vocabulary by repeatedly merging frequent adjacent symbols.

BPE has two main phases:

1. Count token pair frequencies.
2. Merge the most frequent pair into a new token.

### A Simple BPE Example

Suppose a corpus contains the following word frequencies:

```text
hug   10
pug    5
pun   12
bun    4
hugs   5
```

The initial split may treat each character as a token:

```text
h u g
p u g
p u n
b u n
h u g s
```

Then BPE counts adjacent pairs:

```text
h u
u g
p u
u n
b u
g s
```

If `u g` is the most frequent pair, BPE merges it into a new token:

```text
u g -> ug
```

The corpus representation becomes:

```text
h ug
p ug
p u n
b u n
h ug s
```

The algorithm repeats this process until the vocabulary reaches the desired size.

### Why BPE Works Well

BPE naturally keeps frequent patterns as larger tokens while allowing rare words to be decomposed.

It also works well for names, domain terms, and morphological variants because shared pieces can be reused across words.

## Byte-Level BPE

Standard BPE needs a base vocabulary. If the base vocabulary contains every possible Unicode character, the initial vocabulary can become large.

Byte-level BPE solves this by using bytes as the base tokens. Since a byte has only 256 possible values, the base vocabulary is small and universal.

This makes byte-level BPE useful for:

- multilingual text
- Chinese, Japanese, Arabic, and other scripts
- emoji
- noisy web text
- rare symbols
- code and mixed-format data

GPT-2 popularized byte-level BPE with a vocabulary of 50,257 tokens: 256 byte-level base tokens, one end-of-text token, and 50,000 learned merge tokens.

The advantage is coverage. The tokenizer can represent almost any text without needing a massive Unicode character vocabulary.

## WordPiece

WordPiece is similar to BPE, but it uses a different merge criterion.

It is commonly associated with BERT-style tokenizers. In BERT, continuation subwords are marked with the `##` prefix:

```text
word -> w, ##o, ##r, ##d
```

or after training:

```text
tokenization -> token, ##ization
```

### How WordPiece Chooses Merges

BPE usually merges the most frequent adjacent pair. WordPiece instead scores a pair using a criterion similar to normalized association:

```text
score(pair) = frequency(pair) / (frequency(token_1) * frequency(token_2))
```

This means WordPiece does not simply favor pairs that appear often. It favors pairs whose combined occurrence is strong relative to how often the individual tokens appear separately.

For example, a pair such as:

```text
un, ##able
```

may be frequent, but `un` and `##able` may also appear in many other words. WordPiece may prefer a more specific pair if the individual pieces are less broadly distributed.

This helps the tokenizer learn useful subword units instead of always merging the most common pieces.

## SentencePiece

SentencePiece is a tokenizer framework rather than a single algorithm. It can train subword models such as BPE and Unigram directly from raw text.

A key difference is that SentencePiece does not require whitespace-based pre-tokenization. This is important for languages where word boundaries are not marked by spaces, and it also makes preprocessing more uniform across languages.

SentencePiece commonly treats whitespace as a normal symbol. This allows the tokenizer to reconstruct the original text more reliably.

SentencePiece is widely used in multilingual and sequence-to-sequence models. Algorithms such as Unigram are often trained through SentencePiece.

## Unigram Tokenization

Unigram tokenization takes a different approach from BPE.

BPE starts with small units and repeatedly adds merged tokens. Unigram starts with a large candidate vocabulary and repeatedly removes tokens.

The process is roughly:

1. Build an initial vocabulary with many possible subword tokens.
2. Estimate token probabilities from the corpus.
3. Compute the likelihood of the corpus under the current vocabulary.
4. Try removing candidate tokens.
5. Remove the tokens that increase loss the least.
6. Repeat until the vocabulary reaches the target size.

The basic assumption is that each token is generated independently:

```text
P(t1, t2, ..., tN) = P(t1) * P(t2) * ... * P(tN)
```

The training objective is usually expressed as negative log-likelihood:

```text
loss = -sum(log P(tokens))
```

### How Unigram Segments Text

For a word such as:

```text
hug
```

there may be multiple possible segmentations:

```text
h, u, g
hu, g
h, ug
hug
```

Unigram assigns probabilities to tokens and chooses the segmentation with the highest probability.

Naively checking every possible segmentation is expensive. In practice, dynamic programming algorithms such as Viterbi are used to find the best segmentation efficiently.

### Why Unigram Is Useful

Unigram provides a probabilistic view of tokenization. Instead of committing only to deterministic frequency merges, it can evaluate how useful each token is for explaining the corpus.

It is used in models and tokenizer pipelines such as ALBERT, T5, mBART, BigBird, and XLNet through SentencePiece-style tokenization workflows.

## Comparing the Main Tokenizers

| Tokenizer | How It Builds Vocabulary | Strengths | Tradeoffs |
| --- | --- | --- | --- |
| Word-based | each word is a token | intuitive, semantically meaningful | huge vocabulary, unknown words |
| Character-based | each character is a token | strong coverage, small English vocabulary | long sequences, weaker semantic units |
| BPE | repeatedly merges frequent pairs | simple, effective, reusable subwords | depends on initial symbols and merge rules |
| Byte-level BPE | applies BPE over bytes | excellent Unicode and noisy-text coverage | some tokens may look less interpretable |
| WordPiece | merges pairs using normalized association | strong for BERT-style models | requires careful training and continuation markers |
| Unigram | prunes a large vocabulary by likelihood | probabilistic, flexible, SentencePiece-friendly | training and decoding are more complex |

## Engineering Tradeoffs in LLM Tokenization

Tokenizer choice affects several practical metrics.

### Vocabulary Size

A larger vocabulary can represent more words directly, but it increases the embedding matrix size.

The embedding matrix shape is roughly:

```text
vocabulary_size x hidden_size
```

If the vocabulary grows, model parameters grow too.

### Sequence Length

A smaller vocabulary usually means more tokens per document. More tokens increase attention cost and inference latency.

This matters because Transformer attention cost grows with sequence length, and long prompts also increase KV cache memory during inference.

### Multilingual Coverage

For multilingual systems, tokenization must handle scripts beyond English. Byte-level BPE and SentencePiece-style training are often better suited for broad coverage.

### Domain Adaptation

In specialized domains, default tokenizers may split important terms poorly. Medical terms, product IDs, code symbols, financial abbreviations, and mixed-language entities can become fragmented.

When building domain-specific models, inspect how the tokenizer handles real business text before training.

## Practical Guidelines

When choosing or evaluating a tokenizer, ask these questions:

- Does it represent your target languages without excessive fragmentation?
- Does it handle rare words, names, IDs, and domain terms well?
- How many tokens does it produce for representative documents?
- Does the vocabulary size fit your model and memory budget?
- Does it preserve enough information for downstream tasks?
- Is it compatible with the base model you plan to fine-tune?

For most LLM work, you should not change the tokenizer casually. The tokenizer is tied to the model's embedding layer and pretraining. If you change the tokenizer, you usually need to resize embeddings and carefully handle continued training.

For retrieval, classification, and domain-specific pretraining, tokenizer analysis is still worth doing. Token length distribution, unknown-token behavior, and fragmentation can explain many downstream performance issues.

## Key Takeaways

Tokenization is the bridge between natural language and Transformer computation.

The main lessons are:

- Transformers consume token IDs, not raw text.
- Word-based tokenization is intuitive but struggles with vocabulary size and unknown words.
- Character-based tokenization has strong coverage but creates long sequences.
- Subword tokenization balances vocabulary size, sequence length, and rare-word coverage.
- BPE learns frequent merged units.
- Byte-level BPE improves Unicode and noisy-text coverage by starting from bytes.
- WordPiece uses a normalized merge score and continuation markers such as `##`.
- SentencePiece trains tokenizers directly from raw text and is useful for multilingual settings.
- Unigram starts from a large candidate vocabulary and prunes tokens using likelihood.

Tokenizer design is not a small implementation detail. It is a core modeling decision that affects training cost, inference efficiency, and model quality.
