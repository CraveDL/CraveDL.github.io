---
layout: default
title: Home
---

<section class="hero">
  <div class="hero-copy">
    <p class="eyebrow">NLP / LLM Algorithm Engineer</p>
    <h1>Building practical AI systems for real business workflows.</h1>
    <p class="lead">
      I work on RAG, Agentic RAG, multimodal document understanding, knowledge graphs,
      information retrieval, and intelligent customer service systems across finance,
      insurance, e-commerce, and enterprise AI.
    </p>
    <div class="hero-actions">
      <a class="button primary" href="/blog/">Read the blog</a>
      <a class="button secondary" href="/about/">About me</a>
    </div>
  </div>
  <img class="avatar" src="https://github.com/CraveDL.png" alt="Jiashu Xu GitHub avatar">
</section>

<section class="metrics" aria-label="Selected project impact">
  <div>
    <strong>95%</strong>
    <span>financial table extraction accuracy</span>
  </div>
  <div>
    <strong>78% to 92%</strong>
    <span>financial RAG retrieval accuracy</span>
  </div>
  <div>
    <strong>193K</strong>
    <span>entities in an e-commerce knowledge graph</span>
  </div>
</section>

<section class="section-grid">
  <article>
    <h2>Technical Focus</h2>
    <ul>
      <li>Production RAG and Agentic RAG workflows</li>
      <li>Embedding optimization and retrieval evaluation</li>
      <li>OCR, VLM, table understanding, and financial document extraction</li>
      <li>Knowledge graph schema design, entity linking, and query understanding</li>
    </ul>
  </article>
  <article>
    <h2>Recent Writing</h2>
    {% assign recent_posts = site.posts | slice: 0, 3 %}
    {% if recent_posts.size > 0 %}
      <ul class="post-list compact">
        {% for post in recent_posts %}
          <li>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
          </li>
        {% endfor %}
      </ul>
    {% else %}
      <p>Notes on RAG, NLP systems, and multimodal AI engineering will appear here.</p>
    {% endif %}
  </article>
</section>

