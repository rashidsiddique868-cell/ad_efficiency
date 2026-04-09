import random
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json

torch.manual_seed(42)
np.random.seed(42)

# ============================================
# TYPED MODELS (OpenEnv requirement)
# ============================================

class Observation(BaseModel):
    user_age: float
    user_tech_interest: float
    user_shopping_interest: float
    available_ads: List[str]
    current_budget: float
    impressions_shown: int
    current_ctr: float
    step_number: int
    context: str

class Action(BaseModel):
    selected_ad: str
    bid_amount: float
    reasoning: Optional[str] = None

class Reward(BaseModel):
    value: float
    click_reward: float
    efficiency_reward: float
    budget_penalty: float
    explanation: str

# ============================================
# AD AUCTION ENVIRONMENT
# ============================================

AD_TYPES = ["tech_ad", "shopping_ad", "general_ad"]

AD_DESCRIPTIONS = {
    "tech_ad":      "Technology product advertisement (laptops, phones, gadgets)",
    "shopping_ad":  "E-commerce shopping advertisement (clothes, accessories)",
    "general_ad":   "General interest advertisement (food, lifestyle, services)",
}

class AdAuctionEnvironment:
    def __init__(self, task_level: str = "easy", max_steps: int = 10):
        self.task_level  = task_level
        self.max_steps   = max_steps
        self.reset()

    def reset(self) -> Observation:
        self.step_number       = 0
        self.total_reward      = 0.0
        self.budget            = 100.0
        self.impressions_shown = 0
        self.clicks            = 0
        self.history           = []
        self.done              = False

        self.user = self._generate_user()
        return self._get_observation()

    def _generate_user(self) -> Dict:
        if self.task_level == "easy":
            tech     = random.uniform(0.7, 1.0)
            shopping = random.uniform(0.0, 0.3)
            age      = random.uniform(0.2, 0.4)
        elif self.task_level == "medium":
            tech     = random.uniform(0.3, 0.7)
            shopping = random.uniform(0.3, 0.7)
            age      = random.uniform(0.3, 0.7)
        else:
            tech     = random.uniform(0.0, 1.0)
            shopping = random.uniform(0.0, 1.0)
            age      = random.uniform(0.0, 1.0)

        return {
            "age":      round(age, 2),
            "tech":     round(tech, 2),
            "shopping": round(shopping, 2),
        }

    def _get_observation(self) -> Observation:
        ctr = self.clicks / max(self.impressions_shown, 1)

        if self.user["tech"] > 0.6:
            context = "User shows strong interest in technology products"
        elif self.user["shopping"] > 0.6:
            context = "User shows strong interest in shopping and fashion"
        else:
            context = "User has balanced mixed interests"

        return Observation(
            user_age               = self.user["age"],
            user_tech_interest     = self.user["tech"],
            user_shopping_interest = self.user["shopping"],
            available_ads          = AD_TYPES,
            current_budget         = round(self.budget, 2),
            impressions_shown      = self.impressions_shown,
            current_ctr            = round(ctr, 3),
            step_number            = self.step_number,
            context                = context,
        )

    def _calculate_click_probability(self, ad: str) -> float:
        tech     = self.user["tech"]
        shopping = self.user["shopping"]

        if ad == "tech_ad":
            base_prob = tech * 0.8 + (1 - shopping) * 0.2
        elif ad == "shopping_ad":
            base_prob = shopping * 0.8 + (1 - tech) * 0.2
        else:
            base_prob = 0.3 + (1 - abs(tech - shopping)) * 0.2

        noise = random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, base_prob + noise))

    def step(self, action: Action):
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        selected_ad  = action.selected_ad
        bid_amount   = max(0.0, min(action.bid_amount, self.budget))

        click_prob   = self._calculate_click_probability(selected_ad)
        clicked      = random.random() < click_prob

        if clicked:
            self.clicks += 1

        self.impressions_shown += 1
        self.budget            -= bid_amount
        self.step_number       += 1

        # REWARD FUNCTION
        # Click reward
        click_reward = click_prob * 2.0 if clicked else -0.1

        # Efficiency reward — did we pick the right ad?
        best_ad       = self._get_best_ad()
        correct_match = 1.0 if selected_ad == best_ad else 0.0
        efficiency_reward = correct_match * 1.0

        # Budget penalty — don't overbid
        budget_penalty = -0.5 if bid_amount > 20 else 0.0

        # Partial progress signal — reward improvement in CTR
        old_ctr = (self.clicks - (1 if clicked else 0)) / max(self.impressions_shown - 1, 1)
        new_ctr = self.clicks / self.impressions_shown
        ctr_improvement = (new_ctr - old_ctr) * 2.0

        total_reward = click_reward + efficiency_reward + budget_penalty + ctr_improvement
        total_reward = round(max(-1.0, min(2.0, total_reward)), 3)

        self.total_reward += total_reward
        self.done = (
            self.step_number >= self.max_steps or
            self.budget <= 0
        )

        reward = Reward(
            value             = total_reward,
            click_reward      = round(click_reward, 3),
            efficiency_reward = round(efficiency_reward, 3),
            budget_penalty    = round(budget_penalty, 3),
            explanation       = f"Ad: {selected_ad} | Clicked: {clicked} | Best was: {best_ad} | CTR: {new_ctr:.2%}",
        )

        self.history.append({
            "step":       self.step_number,
            "ad":         selected_ad,
            "bid":        bid_amount,
            "clicked":    clicked,
            "reward":     total_reward,
            "click_prob": round(click_prob, 3),
        })

        obs  = self._get_observation()
        info = {
            "clicked":      clicked,
            "click_prob":   round(click_prob, 3),
            "best_ad":      best_ad,
            "correct_match": bool(correct_match),
            "budget_left":  round(self.budget, 2),
            "total_reward": round(self.total_reward, 3),
        }

        return obs, reward, self.done, info

    def _get_best_ad(self) -> str:
        if self.user["tech"] > 0.6:
            return "tech_ad"
        elif self.user["shopping"] > 0.6:
            return "shopping_ad"
        else:
            return "general_ad"

    def state(self) -> Dict:
        return {
            "task_level":       self.task_level,
            "step_number":      self.step_number,
            "max_steps":        self.max_steps,
            "budget":           round(self.budget, 2),
            "impressions":      self.impressions_shown,
            "clicks":           self.clicks,
            "ctr":              round(self.clicks / max(self.impressions_shown, 1), 3),
            "total_reward":     round(self.total_reward, 3),
            "done":             self.done,
            "user":             self.user,
            "history":          self.history,
        }

# ============================================
# GRADERS (one per task level)
# ============================================

def grade_easy(env: AdAuctionEnvironment) -> float:
    if env.impressions_shown == 0:
        return 0.001
    ctr          = env.clicks / env.impressions_shown
    correct_ads  = sum(1 for h in env.history if h["ad"] == "tech_ad")
    correct_rate = correct_ads / len(env.history) if env.history else 0
    score = (ctr * 0.5) + (correct_rate * 0.5)
    return round(min(0.999, max(0.001, score)), 3)

def grade_medium(env: AdAuctionEnvironment) -> float:
    if env.impressions_shown == 0:
        return 0.001
    ctr            = env.clicks / env.impressions_shown
    budget_used    = (100 - env.budget) / 100
    efficiency     = ctr / max(budget_used, 0.01)
    correct_ads    = sum(1 for h in env.history
                        if h["ad"] == env._get_best_ad())
    correct_rate   = correct_ads / len(env.history) if env.history else 0
    score = (ctr * 0.4) + (correct_rate * 0.4) + (min(efficiency, 1.0) * 0.2)
    return round(min(0.999, max(0.001, score)), 3)

def grade_hard(env: AdAuctionEnvironment) -> float:
    if env.impressions_shown == 0:
        return 0.001
    ctr          = env.clicks / env.impressions_shown
    budget_eff   = env.total_reward / max(env.impressions_shown, 1)
    correct_ads  = sum(1 for h in env.history
                      if h["ad"] == env._get_best_ad())
    correct_rate = correct_ads / len(env.history) if env.history else 0
    avg_reward   = env.total_reward / max(env.impressions_shown, 1)
    norm_reward  = min(1.0, max(0.0, (avg_reward + 1) / 3))
    score = (
        ctr          * 0.3 +
        correct_rate * 0.3 +
        norm_reward  * 0.2 +
        min(1.0, env.budget / 100) * 0.2
    )
    return round(min(0.999, max(0.001, score)), 3)

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}

def run_episode(task_level="easy", agent_fn=None) -> Dict:
    env  = AdAuctionEnvironment(task_level=task_level)
    obs  = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if agent_fn:
            action = agent_fn(obs, env.state())
        else:
            best = env._get_best_ad()
            action = Action(
                selected_ad = best,
                bid_amount  = 10.0,
                reasoning   = "Baseline: always pick best matching ad",
            )

        obs, reward, done, info = env.step(action)
        total_reward += reward.value

    score = GRADERS[task_level](env)
    return {
        "task_level":   task_level,
        "score":        score,
        "total_reward": round(total_reward, 3),
        "clicks":       env.clicks,
        "impressions":  env.impressions_shown,
        "ctr":          round(env.clicks / max(env.impressions_shown, 1), 3),
        "history":      env.history,
    }

if __name__ == "__main__":
    print("Testing AdAuction Environment\n")
    print("=" * 40)

    for level in ["easy", "medium", "hard"]:
        result = run_episode(level)
        print(f"Task: {level.upper()}")
        print(f"  Score:       {result['score']}")
        print(f"  CTR:         {result['ctr']:.1%}")
        print(f"  Clicks:      {result['clicks']}/{result['impressions']}")
        print(f"  Total Reward:{result['total_reward']}")
        print()

    print("Environment working correctly!")