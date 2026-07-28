---
layout: page
title: Projects
permalink: /projects/
---

Apps, agents, and systems — each with reproducibility links back to the source.

{% assign items = site.projects | sort: "order" %}
<ul class="listing">
  {% for p in items %}
    <li>
      <a href="{{ p.url }}">{{ p.title }}</a>
      {% if p.summary %}<span class="summary">{{ p.summary }}</span>{% endif %}
    </li>
  {% endfor %}
</ul>
