---
layout: post
title: "Building Practical RAG Systems: Notes from Production AI Workflows"
date: 2026-05-25
categories: [RAG, LLM, NLP]
---

RAG systems look simple in demos: split documents, build embeddings, retrieve context, and ask an LLM to answer. In production, the hard parts are usually less glamorous. The quality of the system depends on document structure, data cleaning, chunking strategy, retrieval evaluation, answer constraints, and continuous error analysis.

## Start with the Data Boundary

Before optimizing models, define what the system is allowed to know. In finance and insurance scenarios, source documents often include reports, policies, product descriptions, FAQs, tables, scanned PDFs, and business rules. Each type has a different structure and failure mode.

A useful first step is to classify documents by structure:

- long narrative text
- FAQ-style question and answer pairs
- policy or contract clauses
- tables and semi-structured records
- scanned or image-based pages
- documents with multi-level headers

The retrieval strategy should follow the document structure, not the other way around.

## Chunking Is a Business Decision

Chunking is often treated as a fixed text processing step. In practice, it determines what evidence the model can see. For insurance QA, splitting blindly by token count can separate product names, coverage responsibilities, exclusions, and conditions that should stay together.

Domain-aware chunking improves retrieval because the chunk carries enough business meaning. For high-risk QA, I prefer chunks that preserve:

- title and section hierarchy
- product or policy identity
- conditions and exceptions
- table headers and row context
- cross-references that affect the answer

Good chunks reduce both missed recall and hallucinated answers.

## Evaluate Retrieval Separately

If the final answer is wrong, the generator may not be the root cause. The retrieval stage might have missed the right evidence or retrieved a similar but incorrect clause.

I usually separate evaluation into at least two layers:

- retrieval accuracy: did the system retrieve the right evidence
- answer accuracy: did the final response use the evidence correctly

This separation makes optimization clearer. For example, embedding optimization with hard negative mining and contrastive learning can improve retrieval before any prompt or fine-tuning work begins.

## Use Constraints for High-Risk Answers

For insurance, finance, and other sensitive domains, prompt engineering is not just about making the answer fluent. It should constrain the model's behavior.

Useful constraints include:

- answer only from retrieved evidence
- state uncertainty when evidence is incomplete
- preserve numbers, dates, and named entities exactly
- avoid unsupported recommendations
- return structured outputs when downstream systems need parsing

These constraints make answers easier to evaluate and safer to deploy.

## Agentic RAG Helps When the Workflow Has Steps

Agentic RAG is valuable when the system needs more than one retrieval call. A practical workflow may include intent recognition, query rewriting, entity extraction, retrieval, answer generation, and result validation.

The key is not to add agents for complexity. The key is to make each step observable and measurable. If a step cannot be evaluated, it will be difficult to improve.

## Production RAG Is an Iteration Loop

The most important habit is continuous error analysis. Collect bad cases, label the root cause, and decide whether the fix belongs in data cleaning, chunking, retrieval, prompt constraints, fine-tuning, or product logic.

RAG quality usually improves through many small, measurable changes rather than one large model change.

