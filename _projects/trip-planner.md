---
layout: project
title: AI Trip Planner
summary: Phase-based build of a travel planning assistant (MVP → auth → memory → eval → ...)
order: 1
repo: https://github.com/abhisheksharma026/Trip-Planner
stack: Python, Google ADK
status: Active
---

## What it does ?
A travel planning assistant that evolves in clear phases (MVP to production-grade features).

## Architecture


## Phase timeline:
- Phase I — MVP travel planner (simple flow, minimal state)

## Reproduce
Need to tag each phase in the code repository, then add instructions

```bash
git clone https://github.com/abhisheksharma026/Trip-Planner
git checkout v1.0.0
```

---
layout: project
title: AI Trip Planner 
summary: An AI trip-planning agent with a modular multi-agent architecture (concierge orchestrator + specialized agents), a CLI entrypoint, and a web UI.
order: 1
repo: https://github.com/abhisheksharma026/Trip-Planner
stack: Python, Google ADK, Flask web UI, Modular agents + tools
status: Active
---

## Overview

**Trip Planner AI** is a modular multi-agent system for intelligent trip planning. The core architecture uses a **Concierge (root) orchestrator** that coordinates specialized agents (e.g., flights, hotels, finance) with a supporting layer for sessions, tools, and execution.  
The implementation ships with both a **CLI** (`main.py`) and a **web interface** (`app.py` + `templates/` + `static/`).

---

## Architecture (high level)

The codebase is structured around:

- **Concierge / root orchestrator**: delegates tasks and coordinates multi-agent flow
- **Specialized agents**: separate responsibilities (e.g., flight recommender, hotel specialist, financial planner)
- **Tools layer**: utilities such as geolocation and itinerary export
- **Session handling**: retains session context for conversational interaction
- **Interfaces**:
  - Web app: `app.py` + `templates/` + `static/`
  - CLI: `main.py`

---

## How to read the project

I document the build in two layers:

1. **Minimal narrative (this site)**  
   Phase summaries and key design decisions.

2. **Implementation notes (main repo)**  
   Deep-dive Markdown notes live inside the Trip-Planner repository under:

- Blog / build notes:  
  https://github.com/abhisheksharma026/Trip-Planner/tree/main/blog

- Setup and usage guides:  
  https://github.com/abhisheksharma026/Trip-Planner/blob/main/SETUP.md  
  https://github.com/abhisheksharma026/Trip-Planner/blob/main/WEB_APP_GUIDE.md

---

## Current release checkpoint

The project currently has a tagged release:

- Tag: **v1.0.0**
- Tag notes describe the initial release and major features
- Commit: **ef6b78d**

Tags:  
https://github.com/abhisheksharma026/Trip-Planner/tags

> Note: The Phase I documentation and blog posts were written after the `v1.0.0` tag was created, and retrospectively describe the architecture and behavior of the system at that release.


---

## Phase plan:

To keep the public narrative clean, I plan to keep the evolution within **up to five phases**, where each phase is a meaningful architectural step (not a micro-update).  
Detailed implementation notes remain in the main repository.

Proposed phases:
1. Phase I — MVP architecture + agents + orchestration + UI + baseline observability
2. Phase II — WIP
3. Phase III — TBD
4. Phase IV — TBD
5. Phase V — TBD

---

## Reproducibility (next step)

Right now the stable checkpoint is `v1.0.0`.  
As the project evolves, I will add **phase tags** (e.g., `phase-1-mvp`, `phase-2-auth`) so readers can check out the exact system state per phase.

```bash
git clone https://github.com/abhisheksharma026/Trip-Planner
git checkout v1.0.0
```
