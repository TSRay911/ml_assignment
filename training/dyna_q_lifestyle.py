"""
  pip install numpy tabulate gymnasium
  python training/dyna_q_lifestyle.py --model training/saved_models/dyna_q_final_best.json --seed 999 --show_summary
"""
import os
import json
import argparse
from typing import Dict, Tuple, List
from collections import Counter
import numpy as np
from tabulate import tabulate
from training.environment3 import LifeStyleEnv


class DynaQLifestyle:

    # ------- Action name mapping  -------
    ACTION_NAME = {
        0: "meal: light", 1: "meal: medium", 2: "meal: heavy",
        3: "exercise: light", 4: "exercise: moderate", 5: "exercise: intense",
        6: "rest: light", 7: "rest: deep", 8: "skip",
    }

    # ------- State binning  -------
    @staticmethod
    def _bin5(x: float, lo: float = 0.0, hi: float = 100.0) -> int:
        x = float(x)
        x = max(lo, min(hi, x))
        width = (hi - lo) / 5.0 if hi > lo else 1.0
        return min(4, int((x - lo) / width))

    @staticmethod
    def _bin_bmi(bmi: float) -> int:
        b = float(bmi)
        if b < 16:  return 0
        if b < 18.5:return 1
        if b < 22:  return 2
        if b < 25:  return 3
        if b < 27:  return 4
        if b < 30:  return 5
        if b < 35:  return 6
        return 7  # >= 35

    @classmethod
    def state_key(cls, obs: Dict) -> Tuple[int, int, int, int, int]:
       
        bmi    = float(obs["current_bmi"][0])
        stress = float(obs["current_stress_level"][0])
        energy = float(obs["current_energy_level"][0])
        hunger = float(obs["current_hunger_level"][0])
        tslot  = int(obs["current_timeslot"])       
        tbin   = tslot // 3                         
        return (cls._bin_bmi(bmi),
                cls._bin5(stress),
                cls._bin5(energy),
                cls._bin5(hunger),
                tbin)

    # ------- Mask helper (optionally cap skip) -------
    @staticmethod
    def get_valid_mask(env: LifeStyleEnv) -> np.ndarray:
       
        if hasattr(env, "action_masks"):
            m = env.action_masks()
        elif hasattr(env, "get_action_mask"):
            m = env.get_action_mask()
        else:
            m = np.ones(env.action_space.n, dtype=bool)

        # Optional policy rule: at most 3 skips/day if env exposes counter
        try:
            if getattr(env, "skip_count", 0) >= 3:
                m = np.asarray(m, dtype=bool)
                if len(m) >= 9:
                    m[8] = False
        except Exception:
            pass

        return np.asarray(m, dtype=bool)

    # ------- Q-table I/O -------
    @staticmethod
    def load_q(path: str) -> Dict[Tuple, float]:
        
        with open(path, "r") as f:
            data = json.load(f)
        # Convert stringified tuples back into tuples
        return {eval(k): v for k, v in data.items()}

    # ------- Greedy action picker (mask-aware) -------
    @classmethod
    def pick_greedy(cls, Q: Dict[Tuple, float], s: Tuple[int, ...], env: LifeStyleEnv) -> int:
        mask = cls.get_valid_mask(env)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0  # safe fallback
        return int(max(idx, key=lambda a: Q.get((s, int(a)), 0.0)))

    # ------- Run one full episode & collect rows -------
    @classmethod
    def run_episode(cls, Q: Dict[Tuple, float], env, seed: int = 123) -> Tuple[List[List], float, Counter]:
      
        obs, info = env.reset(seed=seed)
        s = cls.state_key(obs)
        done = False
        rows: List[List] = []
        total_reward = 0.0
        a_hist = Counter()

        while not done:

            mask = cls.get_valid_mask(env)
            idx = np.where(mask)[0]
            a = int(max(idx, key=lambda aa: Q.get((s, int(aa)), 0.0))) if len(idx) else 0

            obs2, r, term, trunc, info = env.step(a)
            
            total_reward += float(r)
            a_hist[a] += 1

            # Pretty logging 
            day = int(obs.get("day_of_episode", 0))
            tslot = int(obs["current_timeslot"])

            rows.append([
                day,
                tslot,
                cls.ACTION_NAME.get(a, str(a)),
                env.env.daily_schedule[tslot],
                f"{float(r):.2f}",
                f"{float(obs['current_bmi'][0]):.2f}",
                f"{float(obs['current_stress_level'][0]):.2f}",
                f"{float(obs['current_energy_level'][0]):.2f}",
                f"{float(obs['current_hunger_level'][0]):.2f}",
                f"{float(obs['daily_calories_intake']):.2f}",
                f"{float(obs['daily_calories_burned']):.2f}"
            ])

            s, obs, done = cls.state_key(obs2), obs2, bool(term or trunc)

        env.close()
        return rows, total_reward, a_hist

    # ------- Render as ASCII table -------
    @staticmethod
    def to_table(rows: List[List]) -> str:
        headers = [
            "Day", "Timeslot", "Action", "Event", "BMI", "Stress",
            "Energy", "Hunger", "Cal. Intake", "Cal. Burned", "Reward"
        ]
        return tabulate(rows, headers=headers, tablefmt="pretty")
    



def main():
    parser = argparse.ArgumentParser(description="Run Dyna-Q Lifestyle best model (inference only).")
    parser.add_argument("--model", type=str, default="saved_models/dyna_q_final_best.json",
                        help="Path to the saved Q-table JSON.")
    parser.add_argument("--seed", type=int, default=999, help="Episode seed.")
    parser.add_argument("--show_summary", action="store_true",
                        help="Print summary (total reward and action usage).")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # Load Q-table
    Q = DynaQLifestyle.load_q(args.model)

    # Run one episode (greedy, mask-aware)
    print("Starting Final Evaluation...\n")
    rows, total_reward, a_hist = DynaQLifestyle.run_episode(Q, seed=args.seed)

    print(DynaQLifestyle.to_table(rows))

    if args.show_summary:
        total_steps = sum(a_hist.values())
        skip_ratio = a_hist.get(8, 0) / max(total_steps, 1)
        print("\n— Summary —")
        print(f"Total steps: {total_steps}")
        print(f"Total return (sum of step rewards): {total_reward:.2f}")
        print(f"Action usage: {dict(sorted(a_hist.items()))}")
        print(f"Skip ratio: {skip_ratio:.2%}")


if __name__ == "__main__":
    main()
