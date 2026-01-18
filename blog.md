---
layout: page
title: Blog
permalink: /blog/
---

Engineering notes and phase-based build logs.

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small> — {{ post.date | date: "%Y-%m-%d" }}</small>
      {% if post.excerpt %}<div>{{ post.excerpt | strip_html | truncate: 220 }}</div>{% endif %}
    </li>
  {% endfor %}
</ul>
