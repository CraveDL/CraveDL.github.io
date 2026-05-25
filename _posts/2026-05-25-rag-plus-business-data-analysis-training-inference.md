---
layout: post
title: "RAG+ for Business Data Analysis: Training and Inference Architecture"
date: 2026-05-25
categories: [RAG, LLM, Retrieval, NLP]
---

RAG is often introduced as a simple pattern: retrieve relevant context, pass it to a large language model, and generate an answer. That pattern is useful, but it is not enough for many business data analysis scenarios.

In real enterprise workflows, questions may involve relational tables, graph relationships, Excel files, HTML pages, scanned PDFs, long text documents, audio, video, and fast-changing domain knowledge. A practical RAG+ system has to combine retrieval, information extraction, structured query generation, model fine-tuning, prompt design, and inference optimization.

This article summarizes a production-oriented RAG+ architecture for business data analysis, with a focus on how to think about data sources, training pipelines, and inference-time retrieval.

## What RAG+ Means

RAG+ extends the classic retrieval-augmented generation pipeline with additional capabilities around data processing, model adaptation, and inference optimization.

Classic RAG usually focuses on three modules:

1. A retrieval module that finds relevant context.
2. A generation module that uses an LLM to produce an answer.
3. An orchestration layer that connects the user query, retriever, context, and model.

RAG+ adds more engineering around the edges:

- schema understanding for structured databases
- information extraction for semi-structured and unstructured documents
- hybrid retrieval across vector search, lexical search, and knowledge graphs
- fine-tuned embedding models for domain-specific recall
- PEFT or full-parameter fine-tuning for answer style and task behavior
- reranking, context compression, and prompt constraints
- inference acceleration with KV cache management and serving frameworks such as vLLM

The goal is not to make the architecture more complicated. The goal is to make it fit the data and the business question.

## Start from the Business Data Type

The most important design decision is not which model to use. It is how the business data should be matched.

Different data types require different retrieval and reasoning strategies.

| Data Type | Examples | Matching Goal | Common Techniques | Main Challenge |
| --- | --- | --- | --- | --- |
| Structured data | relational databases, graph databases, data warehouses | exact matching | SQL, Cypher, Spark SQL, schema linking | generating correct queries and executing them efficiently |
| Semi-structured data | Excel, XML, HTML, forms, tables | exact or field-level matching | layout analysis, information extraction, triple extraction | understanding structure, headers, entities, and relationships |
| Unstructured data | PDF, plain text, audio, video | similarity matching | OCR, ASR, object detection, text embedding, semantic search | handling diverse formats and long-tail user questions |

A strong RAG+ system often uses all three.

## Structured Data: Exact Matching and Query Generation

Structured data is usually best handled through exact matching. A user may ask a natural language question, but the system eventually needs to map that question to entities, relations, attributes, filters, and query logic.

For simple questions, this may be a single-table lookup. For complex questions, the system may need multi-hop reasoning across tables or graph nodes.

The core technical challenge is query generation:

- understand the database schema
- identify entities, attributes, and relationships
- decide whether the question is single-hop or multi-hop
- generate SQL, Cypher, or Spark SQL
- execute the query within acceptable latency
- validate whether the result actually answers the question

LLMs can generate structured queries, but direct generation is fragile when the schema is large or the entity relationships are subtle. A safer pattern is to decompose the workflow into steps:

1. classify the question type
2. identify entities and candidate tables or graph labels
3. retrieve schema snippets relevant to the question
4. generate a query
5. run query validation
6. execute and summarize the result

This gives the system more control and makes failures easier to debug.

## Semi-Structured Data: Turn Layout into Structure

Semi-structured data sits between databases and free text. Excel files, HTML pages, XML documents, product tables, and form-like documents contain structure, but that structure is not always clean or consistent.

The goal is often to convert semi-structured content into a more explicit representation before retrieval. For example:

- extract entities and attributes
- identify table headers and row relationships
- normalize values and units
- convert facts into triples
- store extracted knowledge in a database or graph

This is where information extraction becomes part of the RAG+ pipeline.

For table-heavy documents, the system has to understand both layout and semantics. A cell value is not meaningful by itself; it depends on row headers, column headers, merged cells, page context, and sometimes nearby text. In financial, insurance, and enterprise documents, this is often the difference between a correct answer and a plausible but wrong one.

## Unstructured Data: Similarity Matching and Long-Tail Questions

Unstructured data is where classic RAG is most visible. Documents are chunked, embedded, indexed, retrieved, and passed into an LLM.

But the hard part is long-tail coverage. Training data often covers common questions well, while rare questions remain poorly represented. For enterprise knowledge bases, the domain of possible questions may be hard to define precisely, even when the knowledge base itself is known.

One practical strategy is to use LLMs to help generate training data:

- generate likely user questions from each document section
- create paraphrases for important intents
- create hard negative pairs for similar but different concepts
- generate domain-specific query-document pairs for embedding fine-tuning

The generated data should still be filtered and evaluated. Synthetic data is useful when it expands coverage, but it can also amplify noise if used carelessly.

## Why Not Use Only Prompt Engineering or Fine-Tuning

LLMs are powerful, but they have several limitations in business systems:

- They can hallucinate and produce unsupported information.
- Their built-in knowledge becomes stale as business knowledge changes.
- Prompt engineering with long context increases token usage and serving cost.
- Fine-tuning can teach behavior, but it does not automatically provide fresh facts.

RAG, fine-tuning, and prompt engineering solve different parts of the problem.

| Technique | Best For | Limitation |
| --- | --- | --- |
| Prompt engineering | controlling answer format and behavior | context grows quickly and may be expensive |
| Fine-tuning | adapting style, task behavior, and domain patterns | does not update external knowledge by itself |
| RAG | grounding answers in current knowledge | depends heavily on retrieval quality |
| Better base models | stronger reasoning and generation | higher cost and still needs grounding |

A practical RAG+ system usually combines them.

## Training the Retrieval Layer

The retrieval layer determines what evidence the LLM can see. If retrieval fails, generation quality has a hard ceiling.

### Chunking Strategy

Chunking should follow document structure and business meaning. Common strategies include:

- sentence-level chunks for short facts
- paragraph-level chunks for coherent explanations
- section-level chunks for policies, reports, and manuals
- table-aware chunks for rows, headers, and surrounding context

The right chunk size depends on the retrieval model, the document type, and the answer task.

### Embedding Fine-Tuning

General-purpose embeddings are often not enough for domain-specific business queries. Fine-tuning models such as BGE can improve recall when the system has high-quality query-document pairs.

Useful training techniques include:

- unsupervised contrastive learning
- hard negative mining
- self-learning and distillation
- long-context retrieval optimization
- unified optimization across multiple retrieval methods
- domain pretraining methods such as RetroMAE-style representation learning

The goal is to teach the embedding model which differences matter in the domain. In finance or insurance, two passages can look semantically similar while implying very different answers.

## Training the Lexical and Keyword Layer

Vector retrieval is not enough for every query. Business questions often contain exact product names, IDs, policy terms, technical entities, locations, or event names.

A keyword retrieval layer can help identify:

- hot words
- domain entities
- product names
- attribute values
- important phrases in user questions

Typical techniques include lexical analysis, maximum matching, Trie-based matching, NER, and rule-enhanced normalization.

In many systems, the best retrieval path is hybrid:

1. vector search for semantic similarity
2. BM25 or TF-IDF for lexical relevance
3. entity matching for exact business terms
4. reranking for final precision

## Fine-Tuning the LLM Layer

LLM fine-tuning can improve answer format, domain language, reasoning pattern, and task consistency. The tradeoff is cost.

Full-parameter fine-tuning updates all model parameters and usually requires much more GPU memory. During training, memory is consumed by:

- model parameters
- gradients
- optimizer states
- intermediate activations

For a model with billions of parameters, this becomes expensive quickly. A 7B model loaded in fp16 already needs roughly 14 GB just for model parameters, before gradients, optimizer states, and activations.

This is why PEFT methods are common in enterprise workflows.

Common PEFT options include:

- prefix-tuning
- P-tuning v1
- P-tuning v2
- LoRA
- QLoRA

PEFT methods reduce the number of trainable parameters while still adapting model behavior. They are especially useful when the goal is task adaptation rather than rebuilding the entire language model.

## Inference Pipeline: Recall First, Then Precision

At inference time, the retrieval pipeline should be designed in layers.

The first stage should protect recall. It is acceptable to retrieve more candidates if the downstream reranker can filter them.

Common recall methods include:

- semantic vector search with Word2Vec, BERT, or BGE-style embeddings
- lexical search with TF-IDF or BM25
- string and entity matching
- hybrid retrieval across multiple indexes

After recall, ranking improves precision.

A common pattern is:

1. retrieve top 5 to 20 candidates from each recall module
2. use a bi-encoder for coarse ranking and efficient scoring
3. use a cross-encoder for fine reranking
4. select the final top contexts for the LLM

Bi-encoders are efficient and good for large-scale candidate scoring. Cross-encoders are slower but more accurate because they jointly encode the query and candidate text. A production system often uses both.

## LLM Inference: KV Cache and Serving Efficiency

LLM inference has two major phases:

1. Pre-fill: process the input prompt and create key-value cache entries for each transformer layer.
2. Decode: generate tokens one by one while reading and updating the KV cache.

KV cache improves inference speed, but it also consumes GPU memory. A simplified fp16 memory estimate is:

```text
KV cache memory ~= 4 * batch_size * num_layers * hidden_size * sequence_length
```

where `sequence_length` includes both input tokens and generated output tokens.

For large models and large batches, KV cache memory can become a major serving bottleneck. This is why serving systems need memory-aware scheduling and cache management.

vLLM's PagedAttention is a practical example. It manages KV cache using a paged memory approach inspired by operating system virtual memory. Instead of requiring large contiguous memory blocks, it stores KV cache in non-contiguous pages, improving memory utilization and throughput.

For RAG systems, this matters because prompts can become long after retrieved context is added. Retrieval quality and context length directly affect inference cost.

## A Practical RAG+ Blueprint

A production-oriented RAG+ system can be organized as the following workflow:

| Stage | Responsibility | Example Techniques |
| --- | --- | --- |
| Data ingestion | collect structured, semi-structured, and unstructured data | database connectors, file parsers, OCR, ASR |
| Data understanding | convert raw data into useful retrieval units | schema extraction, layout analysis, entity extraction |
| Indexing | build retrieval indexes | vector indexes, BM25 indexes, graph indexes |
| Training | adapt retrievers and generators | embedding fine-tuning, contrastive learning, PEFT |
| Query understanding | parse the user's intent | classification, NER, schema linking |
| Retrieval | collect candidate evidence | vector search, keyword search, graph traversal |
| Reranking | improve precision | bi-encoder coarse ranking, cross-encoder reranking |
| Generation | produce grounded answers | prompt constraints, answer formatting, citation logic |
| Serving | optimize latency and cost | KV cache, batching, vLLM, context compression |
| Evaluation | close the improvement loop | recall accuracy, answer accuracy, latency, hallucination rate |

## Key Takeaways

RAG+ is not just "RAG with a bigger model." It is an engineering architecture for grounding LLMs in business data.

The most important lessons are:

- Design retrieval around the data type and matching goal.
- Use exact matching for structured data and semantic matching for unstructured data.
- Convert semi-structured content into explicit facts before retrieval when possible.
- Fine-tune embeddings when domain-specific recall matters.
- Combine vector retrieval, lexical retrieval, entity matching, and reranking.
- Use PEFT when task adaptation is needed but full fine-tuning is too expensive.
- Treat KV cache and context length as first-class serving constraints.
- Evaluate retrieval and generation separately so failures are diagnosable.

The best RAG+ systems are not built by adding one more model. They are built by making every stage observable, measurable, and aligned with the business question.
