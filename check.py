import urllib.request
import json

BASE = "https://11arsalan-ad-auction-env.hf.space"

print("Testing all endpoints...\n")

# 1. Health
r = urllib.request.urlopen(f"{BASE}/health")
h = json.loads(r.read())
print(f"1. /health: {h['status']} - data: {h['data_source']}")

# 2. Tasks
r = urllib.request.urlopen(f"{BASE}/tasks")
t = json.loads(r.read())
print(f"2. /tasks: {len(t['tasks'])} tasks found")

# 3. Reset
req = urllib.request.Request(
    f"{BASE}/reset",
    data=json.dumps({"task": "easy", "session_id": "check"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
r   = urllib.request.urlopen(req)
obs = json.loads(r.read())["observation"]
print(f"3. /reset: OK - {obs['context']}")

# 4. Step
req = urllib.request.Request(
    f"{BASE}/step",
    data=json.dumps({
        "session_id": "check",
        "selected_ad": "tech_ad",
        "bid_amount": 10.0
    }).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
r    = urllib.request.urlopen(req)
step = json.loads(r.read())
print(f"4. /step: reward={step['reward']['value']} done={step['done']}")

# 5. Grade
req = urllib.request.Request(
    f"{BASE}/grade",
    data=json.dumps({"session_id": "check"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
r = urllib.request.urlopen(req)
g = json.loads(r.read())
print(f"5. /grade: score={g['score']} (must be 0.0-1.0)")

# 6. State
r = urllib.request.urlopen(f"{BASE}/state?session_id=check")
s = json.loads(r.read())
print(f"6. /state: step={s['step_number']} budget={s['budget']}")

print("\nAll checks passed! Ready to submit!")