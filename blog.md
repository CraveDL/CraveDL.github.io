---
layout: default
title: Blog
permalink: /blog/
---

<section class="page-heading">
  <p class="eyebrow">Technical Blog</p>
  <h1>Notes on NLP, LLM applications, RAG, and production AI systems.</h1>
</section>

<section>
  <ul class="post-list">
    {% for post in site.posts %}
      <li>
        <a href="{{ post.url }}">{{ post.title }}</a>
        <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %-d, %Y" }}</time>
        {% if post.excerpt %}
          <p>{{ post.excerpt | strip_html | truncate: 180 }}</p>
        {% endif %}
      </li>
    {% endfor %}
  </ul>
</section>

