import os
import sys
import json
import random
from typing import List, Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "dummy"
BENCHMARK    = "AdAuctionEnv"
MAX_STEPS    = 10
SUCCESS_SCORE_THRESHOLD = 0.5

# Initialize client with robust error handling for the validator environment
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception as e:
    print(f"[DEBUG] Client init error: {e}", file=sys.stderr, flush=True)
    # Final fallback to standard OpenAI defaults if custom base_url fails
    client = OpenAI(api_key=API_KEY if API_KEY != "dummy" else "missing-key")

SYSTEM_PROMPT = """You are an expert digital advertising agent optimizing ad auctions.

Your job is to select the best advertisement to show a user to maximize
click-through rate and revenue efficiency.

Available ads:
- tech_ad: Technology products (laptops, phones, gadgets)
- shopping_ad: E-commerce shopping (clothes, accessories)
- general_ad: General interest (food, lifestyle, services)

You must respond with a JSON object ONLY:
{
  "selected_ad": "tech_ad" or "shopping_ad" or "general_ad",
  "bid_amount": a number between 1 and 15,
  "reasoning": "brief explanation"
}

Rules:
- Match the ad to the user interests
- Keep bid_amount under 15 to preserve budget
- Higher interest score means more relevant ad"""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_action(obs: dict) -> dict:
    user_prompt = f"""Current advertising situation:

User Profile:
- Age score: {obs.get('user_age', 0.5):.2f} (0=young, 1=old)
- Tech interest: {obs.get('user_tech_interest', 0.5):.2f} (0=low, 1=high)
- Shopping interest: {obs.get('user_shopping_interest', 0.5):.2f} (0=low, 1=high)
- Context: {obs.get('context', 'Unknown')}

Campaign Status:
- Budget remaining: ${obs.get('current_budget', 100):.2f}
- Impressions shown: {obs.get('impressions_shown', 0)}
- Current CTR: {obs.get('current_ctr', 0):.1%}
- Step: {obs.get('step_number', 0) + 1}

Choose the best ad to show this user."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        content = response.choices[0].message.content
        data    = json.loads(content)
        return {
            "selected_ad": data.get("selected_ad", "general_ad"),
            "bid_amount":  float(data.get("bid_amount", 10.0)),
            "reasoning":   data.get("reasoning", ""),
        }
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", file=sys.stderr, flush=True)
        tech     = obs.get("user_tech_interest", 0.5)
        shopping = obs.get("user_shopping_interest", 0.5)
        if tech > 0.6:
            ad = "tech_ad"
        elif shopping > 0.6:
            ad = "shopping_ad"
        else:
            ad = "general_ad"
        return {"selected_ad": ad, "bid_amount": 10.0, "reasoning": "fallback"}

def run_task(task_level: str) -> dict:
    import urllib.request

    BASE       = os.getenv("SPACE_URL", "https://11arsalan-ad-auction-env.hf.space")
    session_id = f"inference_{task_level}_{random.randint(1000,9999)}"

    def post(endpoint, data):
        req = urllib.request.Request(
            f"{BASE}{endpoint}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        r = urllib.request.urlopen(req, timeout=30)
        return json.loads(r.read())

    def get(endpoint):
        r = urllib.request.urlopen(f"{BASE}{endpoint}", timeout=30)
        return json.loads(r.read())

    log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

    result  = post("/reset", {"task": task_level, "session_id": session_id})
    obs     = result["observation"]
    done    = False
    rewards = []
    steps   = 0
    score   = 0.0
    success = False

    try:
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_action(obs)
            action["session_id"] = session_id

            result = post("/step", action)
            obs    = result["observation"]
            reward = result["reward"]["value"]
            done   = result["done"]
            error  = None

            rewards.append(reward)
            steps = step

            log_step(
                step   = step,
                action = action["selected_ad"],
                reward = reward,
                done   = done,
                error  = error,
            )

        grade   = post("/grade", {"session_id": session_id})
        score   = grade["score"]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", file=sys.stderr, flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(
            success = success,
            steps   = steps,
            score   = score,
            rewards = rewards,
        )

    return {"task": task_level, "score": score}

if __name__ == "__main__":
    results = []
    for task in ["easy", "medium", "hard"]:
        result = run_task(task)
        results.append(result)
        print(f"[DEBUG] Task {task} completed: score={result['score']}", file=sys.stderr, flush=True)

    print(f"[DEBUG] All tasks complete: {results}", file=sys.stderr, flush=True)