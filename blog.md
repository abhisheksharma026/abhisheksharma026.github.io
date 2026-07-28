---
layout: page
title: Blog
permalink: /blog/
---

Engineering notes and phase-based build logs.

<ul class="listing">
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small> — {{ post.date | date: "%Y-%m-%d" }}</small>
      {% if post.excerpt %}<span class="summary">{{ post.excerpt | strip_html | truncate: 220 }}</span>{% endif %}
    </li>
  {% else %}
    <li><span class="summary">First notes are on the way.</span></li>
  {% endfor %}
</ul>
