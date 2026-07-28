---
layout: page
title: Research Notes
permalink: /research/
---

Paper reviews and technical notes — intuition, limitations, and applied relevance.

{% assign items = site.research | sort: "date" | reverse %}
<ul class="listing">
  {% for r in items %}
    <li>
      <a href="{{ r.url }}">{{ r.title }}</a>
      {% if r.paper_year %} <span class="meta">({{ r.paper_year }})</span>{% endif %}
      {% if r.summary %}<span class="summary">{{ r.summary }}</span>{% endif %}
    </li>
  {% else %}
    <li><span class="summary">Notes are on the way.</span></li>
  {% endfor %}
</ul>
