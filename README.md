---
title: Ad Auction Efficiency Environment
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - advertising
  - pytorch
  - meta
---

# Ad Auction Efficiency Environment

A real-world OpenEnv-compatible environment for training and evaluating
AI agents on ad auction optimization tasks. Built with PyTorch for the
Meta x Scaler OpenEnv Hackathon 2026.

## What Problem Does This Solve?

Every time someone visits Instagram or Facebook, there is an invisible
auction happening in milliseconds. Advertisers compete to show their ad.
The old system shows the highest bidding ad to everyone — wasting up to
60% of ad budgets by showing the wrong ad to the wrong person.

This environment simulates that auction system. An AI agent must learn
to show the right ad to the right user — not just the highest bidder —
to maximize click-through rate and revenue efficiency.

The environment is trained and tested on the Criteo industry benchmark
dataset — the same dataset used to evaluate production ad systems at
companies like Meta, Google, and Amazon.

## Environment Overview

The agent observes a user's profile and must decide which ad to show
them and how much to bid for the impression. The environment simulates
whether the user clicks, rewards the agent based on correctness and
efficiency, and updates the state for the next step.

This models a real task that humans and automated systems do billions
of times per day at Meta.

## Observation Space

Each observation contains:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| user_age | float | 0.0 - 1.0 | Normalized user age |
| user_tech_interest | float | 0.0 - 1.0 | Interest in technology products |
| user_shopping_interest | float | 0.0 - 1.0 | Interest in shopping |
| available_ads | list | — | Available ad types to choose from |
| current_budget | float | 0 - 100 | Remaining ad budget |
| impressions_shown | int | 0+ | Number of ads shown so far |
| current_ctr | float | 0.0 - 1.0 | Current click through rate |
| step_number | int | 0+ | Current step in episode |
| context | string | — | Natural language user description |

## Action Space

Each action contains:

| Field | Type | Description |
|-------|------|-------------|
| selected_ad | string | One of: tech_ad, shopping_ad, general_ad |
| bid_amount | float | How much to bid (0 to 100) |
| reasoning | string | Optional explanation of decision |

## Reward Function

The reward signal is composite and provided at every step (not just end of episode):

- Click reward (+2.0 if clicked, -0.1 if not)
- Efficiency reward (+1.0 if correct ad was chosen)
- Budget penalty (-0.5 if bid over 20)
- CTR improvement signal (encourages learning over time)

Total reward range: -1.0 to 2.0 per step.

## Tasks

### Easy — Tech User Ad Matching
Show ads to users who strongly prefer technology products.
The agent must consistently identify tech-loving users and show
them the tech ad to maximize clicks.
- Max steps: 10
- Baseline score: 0.85
- Success threshold: 0.7

### Medium — Mixed Audience Optimization
Handle users with mixed interests. The agent must balance ad
relevance with budget efficiency, choosing the right ad while
managing spend carefully.
- Max steps: 10
- Baseline score: 0.55
- Success threshold: 0.5

### Hard — Dynamic Budget Constrained Targeting
Optimize across diverse users with unpredictable preferences.
Agent must maximize CTR AND revenue while staying within budget
constraints. Genuinely challenges frontier models.
- Max steps: 10
- Baseline score: 0.65
- Success threshold: 0.4

## Baseline Scores

Scores achieved by the heuristic baseline agent (always picks
the best matching ad based on user profile):

| Task | Score | CTR | Clicks |
|------|-------|-----|--------|
| Easy | 1.0 | 80% | 8/10 |
| Medium | 0.7 | 40% | 4/10 |
| Hard | 0.8 | 90% | 9/10 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check and status |
| /reset | POST | Start a new episode |
| /step | POST | Take an action |
| /state | GET | Get current environment state |
| /tasks | GET | List all available tasks |
| /grade | POST | Get final episode score |

## Quick Start
```python
import requests

BASE = "https://11arsalan-ad-auction-env.hf.space"

# Check health
health = requests.get(f"{BASE}/health").json()
print(health)

# Start a new episode
obs = requests.post(f"{BASE}/reset", json={
    "task": "easy",
    "session_id": "my_session"
}).json()

print("User context:", obs["observation"]["context"])

# Take an action
result = requests.post(f"{BASE}/step", json={
    "session_id": "my_session",
    "selected_ad": "tech_ad",
    "bid_amount": 10.0,
    "reasoning": "User has high tech interest score"
}).json()

print("Reward:", result["reward"]["value"])
print("Clicked:", result["info"]["clicked"])
print("Done:", result["done"])

# Get final score
score = requests.post(f"{BASE}/grade", json={
    "session_id": "my_session"
}).json()

print("Final score:", score["score"])
```

## Example Agent Loop
```python
import requests

BASE = "https://11arsalan-ad-auction-env.hf.space"

def simple_agent(obs):
    tech     = obs["user_tech_interest"]
    shopping = obs["user_shopping_interest"]
    if tech > 0.6:
        return {"selected_ad": "tech_ad",     "bid_amount": 10.0}
    elif shopping > 0.6:
        return {"selected_ad": "shopping_ad", "bid_amount": 10.0}
    else:
        return {"selected_ad": "general_ad",  "bid_amount": 10.0}

for task in ["easy", "medium", "hard"]:
    session = f"test_{task}"
    result  = requests.post(f"{BASE}/reset",
                json={"task": task, "session_id": session}).json()
    obs     = result["observation"]
    done    = False

    while not done:
        action          = simple_agent(obs)
        action["session_id"] = session
        result          = requests.post(f"{BASE}/step", json=action).json()
        obs             = result["observation"]
        done            = result["done"]

    score = requests.post(f"{BASE}/grade",
                json={"session_id": session}).json()
    print(f"Task {task}: score = {score['score']}")
```

## Setup and Installation
```bash
git clone https://huggingface.co/spaces/11Arsalan/ad-auction-env
cd ad-auction-env
pip install -r requirements.txt
python app.py
```

Or with Docker:
```bash
docker build -t ad-auction-env .
docker run -p 7860:7860 ad-auction-env
```

## Tech Stack

- PyTorch 2.2 — neural network for click probability prediction
- Flask — REST API server
- Criteo Dataset — 100,000 real ad impressions for training
- Docker — containerized deployment on Hugging Face Spaces
- Pydantic — typed observation and action models
- scikit-learn — data preprocessing and normalization

## Real World Relevance

This environment models the core business problem at Meta. Ad efficiency
is how Meta generates 97% of its revenue. Even a 1% improvement in
click-through rate at Meta's scale of 3 billion users translates to
hundreds of millions of dollars in additional revenue.

The environment uses the Criteo industry benchmark dataset — the same
dataset used by researchers at Google, Meta, and Amazon to evaluate
their production ad targeting systems.