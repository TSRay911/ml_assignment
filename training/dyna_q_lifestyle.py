
"""
Usage:
  python training/dyna_q_lifestyle.py --model training/saved_models/dyna_q_final_best.json --seed 999 --show_summary
"""

import os
import json
from typing import Dict, Tuple, List, Optional
from collections import Counter

import numpy as np
from tabulate import tabulate  
from .environment3 import LifeStyleEnv 

# ===== Inference settings (match your training) =====
MAX_SKIPS_PER_DAY = 2
ACTION_NAME = {
    0: "meal: light", 1: "meal: medium", 2: "meal: heavy",
    3: "exercise: light", 4: "exercise: moderate", 5: "exercise: intense",
    6: "rest: light", 7: "rest: deep", 8: "skip",
}

# ---------- Discretization helpers (same as training) ----------
def _bin5(x: float, lo: float = 0.0, hi: float = 100.0) -> int:
    x = float(x); x = max(lo, min(hi, x))
    width = (hi - lo) / 5.0 if hi > lo else 1.0
    return min(4, int((x - lo) / width))

def _bin_bmi(bmi: float) -> int:
    b = float(bmi)
    if b < 16:  return 0
    if b < 18.5:return 1
    if b < 22:  return 2
    if b < 25:  return 3
    if b < 27:  return 4
    if b < 30:  return 5
    if b < 35:  return 6
    return 7

def state_key(obs) -> Tuple[int, int, int, int, int]:
    bmi    = float(obs["current_bmi"][0])
    stress = float(obs["current_stress_level"][0])
    energy = float(obs["current_energy_level"][0])
    hunger = float(obs["current_hunger_level"][0])
    tslot  = int(obs["current_timeslot"])  # 0..23
    tbin   = tslot // 3
    return (_bin_bmi(bmi), _bin5(stress), _bin5(energy), _bin5(hunger), tbin)

def current_event(env) -> str:
    try:
        tslot = int(env.state["current_timeslot"])
        return env.daily_schedule[tslot] if tslot < len(env.daily_schedule) else "action"
    except Exception:
        return "action"

def get_valid_mask_with_cap(env: LifeStyleEnv, local_skip_count: int) -> np.ndarray:
    if hasattr(env, "action_masks"):
        m = np.asarray(env.action_masks(), dtype=bool)
    elif hasattr(env, "get_action_mask"):
        m = np.asarray(env.get_action_mask(), dtype=bool)
    else:
        m = np.ones(env.action_space.n, dtype=bool)

    event = current_event(env)
    if event in ("action", "work") and local_skip_count >= MAX_SKIPS_PER_DAY and len(m) > 8:
        m[8] = False
    return m

def _scalar(x):
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else x
    return x

# ---------- Public inference class ----------
class DynaQLifestyle:
    """Inference runner for your Dyna-Q Q-table (mask-aware, skip-cap aware)."""

    @staticmethod
    def load_q(path: str) -> Dict[Tuple, float]:
        with open(path, "r") as f:
            data = json.load(f)
      
        return {eval(k): float(v) for k, v in data.items()}

    @staticmethod
    def pick_greedy(Q: Dict[Tuple, float], s: Tuple[int, ...], env: LifeStyleEnv, local_skip_count: int) -> int:
        mask = get_valid_mask_with_cap(env, local_skip_count)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0
        return int(max(idx, key=lambda a: Q.get((s, int(a)), 0.0)))

    @classmethod
    def run_episode(
        cls,
        Q: Dict[Tuple, float],
        *,
        seed: int = 999,
        initial_weight_kg: Optional[float] = None,
        days_per_episode: Optional[int] = None
    ) -> Tuple[List[List], float, Counter]:
        kwargs = {}
        if initial_weight_kg is not None:
            kwargs["initial_weight_kg"] = initial_weight_kg
        if days_per_episode is not None:
            kwargs["days_per_episode"] = days_per_episode

        env = LifeStyleEnv(**kwargs)
        obs, info = env.reset(seed=seed)
        s = state_key(obs)
        done = False
        rows: List[List] = []
        total_reward = 0.0
        a_hist = Counter()
        skip_today = 0

        while not done:
            # reset daily skip counter at midnight
            if int(obs["current_timeslot"]) == 0:
                skip_today = 0

            a = cls.pick_greedy(Q, s, env, skip_today)
            obs2, r, term, trunc, _ = env.step(a)
            total_reward += float(r)
            a_hist[a] += 1
            if a == 8:
                skip_today += 1

            day  = int(_scalar(obs.get("day_of_episode", 0)))
            slot = int(obs["current_timeslot"])
            event = env.daily_schedule[slot] if hasattr(env, "daily_schedule") and slot < len(env.daily_schedule) else ""

            rows.append([
                day, slot, ACTION_NAME.get(a, str(a)), event,
                f"{float(r):.2f}",
                f"{float(obs['current_bmi'][0]):.2f}",
                f"{float(obs['current_stress_level'][0]):.2f}",
                f"{float(obs['current_energy_level'][0]):.2f}",
                f"{float(obs['current_hunger_level'][0]):.2f}",
                f"{float(obs['daily_calories_intake']):.2f}",
                f"{float(obs['daily_calories_burned']):.2f}",
                
            ])

            s, obs, done = state_key(obs2), obs2, bool(term or trunc)

        env.close()
        return rows, total_reward, a_hist


# ---------- CLI entrypoint ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Dyna-Q Lifestyle best model (inference only).")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the saved Q-table JSON (e.g. training/saved_models/dyna_q_final_best.json)")
    parser.add_argument("--seed", type=int, default=999, help="Episode seed.")
    parser.add_argument("--initial_weight", type=float, default=None,
                        help="Optional: override initial weight (kg).")
    parser.add_argument("--days", type=int, default=None,
                        help="Optional: override days per episode.")
    parser.add_argument("--show_summary", action="store_true",
                        help="Print summary (total reward and action usage).")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # Load Q-table
    Q = DynaQLifestyle.load_q(args.model)

    # Run one episode (greedy, mask-aware)
    print("Starting Final Evaluation...\n")
    rows, total_reward, a_hist = DynaQLifestyle.run_episode(
        Q,
        seed=args.seed,
        initial_weight_kg=args.initial_weight,
        days_per_episode=args.days
    )

    # Pretty print table
    headers = [
        "Day","Timeslot","Action","Event","BMI","Stress",
        "Energy","Hunger","Cal. Intake","Cal. Burned","Reward"
    ]
    print(tabulate(rows, headers=headers, tablefmt="pretty"))

    # Optional summary
    if args.show_summary:
        steps = sum(a_hist.values())
        skip_ratio = a_hist.get(8, 0) / max(steps, 1)
        print("\n— Summary —")
        print(f"Total steps: {steps}")
        print(f"Total return (sum of step rewards): {total_reward:.2f}")
        print(f"Action usage: {dict(sorted(a_hist.items()))}")
        print(f"Skip ratio: {skip_ratio:.2%}")
