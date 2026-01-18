---
layout: page
title: Projects
permalink: /projects/
---

Below is a list of projects.

{% assign items = site.projects | sort: "order" %}
<ul>
  {% for p in items %}
    <li>
      <a href="{{ p.url }}">{{ p.title }}</a>
      {% if p.summary %} — {{ p.summary }}{% endif %}
    </li>
  {% endfor %}
</ul>
