---
layout: page
title: Research Notes
permalink: /research/
---

Paper reviews and technical notes.

{% assign items = site.research | sort: "date" | reverse %}
<ul>
  {% for r in items %}
    <li>
      <a href="{{ r.url }}">{{ r.title }}</a>
      {% if r.paper_year %} ({{ r.paper_year }}){% endif %}
      {% if r.summary %} — {{ r.summary }}{% endif %}
    </li>
  {% endfor %}
</ul>
