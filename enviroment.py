import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LifeStyleCoachEnv(gym.Env):

    def __init__(self, initial_weight_kg: float = 65, height_cm: float = 170, age: int = 22,
                 gender: int = 0, target_bmi: float = 20, stress_range: tuple[float, float] = (1.0, 100.0),
                 days_per_episode: int = 28):
        
        super().__init__()
        self.initial_weight_kg = initial_weight_kg
        self.height_cm = height_cm
        self.age = age
        self.gender = gender  
        self.target_bmi = target_bmi
        self.days_per_episode = days_per_episode
        self.min_stress_level, self.max_stress_level = stress_range
        self.state = {}
        self.slots_per_day = 24
        self.daily_schedule = ['agent_controlled_action'] * self.slots_per_day

        # Setup schedule
        for start, end in [(10, 12), (13, 17)]:
            for h in range(start, end):
                self.daily_schedule[h] = 'work'
        for h in range(6):
            self.daily_schedule[h] = 'sleep'
        for h in range(22, 24):
            self.daily_schedule[h] = 'sleep'

        self.daily_targets = {
            "protein": 0.18,
            "fat": 0.26,
            "saturated_fat": 0.05,
            "carbs": 0.51,
            "fiber": 30
        }

        self.observation_space = spaces.Dict({
            "current_timeslot": spaces.Discrete(24),
            "current_bmi": spaces.Box(low=10.0, high=80.0, shape=(), dtype=np.float32),
            "daily_calories_intake": spaces.Box(low=0.0, high=6000.0, shape=(), dtype=np.float32),
            "daily_calories_burned": spaces.Box(low=0.0, high=6000.0, shape=(), dtype=np.float32),
            "daily_protein_intake": spaces.Box(low=0.0, high=300.0, shape=(), dtype=np.float32),
            "daily_fat_intake": spaces.Box(low=0.0, high=150.0, shape=(), dtype=np.float32),
            "daily_saturated_fat_intake": spaces.Box(low=0.0, high=50.0, shape=(), dtype=np.float32),
            "daily_carbs_intake": spaces.Box(low=0.0, high=700.0, shape=(), dtype=np.float32),
            "daily_fiber_intake": spaces.Box(low=0.0, high=60.0, shape=(), dtype=np.float32),
            "current_weight_kg": spaces.Box(low=40.0, high=300.0, shape=(), dtype=np.float32),
            "stress_level": spaces.Box(low=self.min_stress_level, high=self.max_stress_level, shape=(), dtype=np.float32),
            "day_of_episode": spaces.Box(low=0, high=self.days_per_episode, shape=(), dtype=np.int32),
        })

        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(3),
            "nutrients": spaces.Box(low=np.array([0.0]*5, dtype=np.float32),
                                     high=np.array([120.0, 60.0, 8.0, 50.0, 20.0], dtype=np.float32),
                                     dtype=np.float32),
            "exercise_level": spaces.Discrete(3),
            "rest_choice": spaces.Discrete(2)
        })

    def calculate_daily_calories(self):
        weight = self.state.get("current_weight_kg", self.initial_weight_kg)
        return weight * (30 if self.gender == 0 else 28)

    def _calculate_bmi(self):
        weight = self.state.get("current_weight_kg", self.initial_weight_kg)
        return weight / ((self.height_cm / 100) ** 2) if self.height_cm > 0 else 0.0

    def _get_obs(self):
        obs_map = {
            "current_timeslot": self.state["time_of_day_slot"],
            "current_bmi": self._calculate_bmi(),
            "daily_calories_intake": self.state["daily_calories_intake"],
            "daily_calories_burned": self.state["daily_calories_burned"],
            "daily_protein_intake": self.state["daily_protein_intake"],
            "daily_fat_intake": self.state["daily_fat_intake"],
            "daily_saturated_fat_intake": self.state["daily_saturated_fat_intake"],
            "daily_carbs_intake": self.state["daily_carbs_intake"],
            "daily_fiber_intake": self.state["daily_fiber_intake"],
            "current_weight_kg": self.state["current_weight_kg"],
            "stress_level": self.state["stress_level"],
            "day_of_episode": self.state["day_of_episode"]
        }
        obs = {}
        for k, v in obs_map.items():
            if isinstance(v, float):
                obs[k] = np.array(v, dtype=np.float32)
            elif isinstance(v, int):
                obs[k] = np.array(v, dtype=np.int32)
            else:
                obs[k] = v
        return obs

    def _get_info(self):
        return {
            "bmi_progress": self.target_bmi - self._calculate_bmi(),
            "weight_progress_kg": self.initial_weight_kg - self.state["current_weight_kg"],
            "calorie_balance": self.state["daily_calories_intake"] - self.state["daily_calories_burned"],
            "stress_history": list(self.state["stress_history"]),
            "weight_history": list(self.state["weight_history"]),
            "days_remaining": self.days_per_episode - self.state["day_of_episode"]
        }

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self.state.update({
            "last_action_type": None,
            "current_weight_kg": self.initial_weight_kg,
            "time_of_day_slot": 0,
            "day_of_episode": 0,
            "daily_calories_intake": 0.0,
            "daily_calories_burned": 0.0,
            "daily_protein_intake": 0.0,
            "daily_fat_intake": 0.0,
            "daily_carbs_intake": 0.0,
            "daily_saturated_fat_intake": 0.0,
            "daily_fiber_intake": 0.0,
            "stress_level": (self.min_stress_level + self.max_stress_level) / 2,
            "weight_history": [],
            "stress_history": [],
        })

        self.daily_calories_goal = self.calculate_daily_calories()
        self.daily_targets_g = {}

        self.daily_targets_g = { 
            "protein": (self.daily_targets["protein"] * self.daily_calories_goal) / 4, 
            "fat": (self.daily_targets["fat"] * self.daily_calories_goal) / 9, 
            "saturated_fat": (self.daily_targets["saturated_fat"] * self.daily_calories_goal) / 9, 
            "carbs": (self.daily_targets["carbs"] * self.daily_calories_goal) / 4, 
            "fiber": self.daily_targets["fiber"] 
            } 
        
        meals_per_day = 3 

        self.nutrients_target_per_meal = {
            nutrient: total_grams / meals_per_day 
            for nutrient, total_grams in self.daily_targets_g.items()
        } 

        return self._get_obs(), self._get_info()

    def step(self, action):

        reward = 0.0
        current_hour = self.state["time_of_day_slot"]
        event = self.daily_schedule[current_hour]
        nutrient_keys = ["protein", "fat", "saturated_fat", "carbs", "fiber"]
        truncated = False

        action_type = action.get("action_type", 0)

        # Handle sleep/work
        if event == "sleep":

            self.state["stress_level"] = max(self.min_stress_level, self.state["stress_level"] - 2)
            self.state["daily_calories_burned"] += (0.9 * self.state["current_weight_kg"] * 3.5 / 200) * (8*60)

        elif event == "work":

            self.state["stress_level"] = min(self.max_stress_level, self.state["stress_level"] + 4)
            self.state["daily_calories_burned"] += (2.0 * self.state["current_weight_kg"] * 3.5 / 200) * 60

        else:

            # Penalize repeated action
            if self.state.get("last_action_type") == action_type:
                reward -= 0.5

            self.state["last_action_type"] = action_type

            if action_type == 0:
                nutrients = action.get("nutrients", np.zeros(5, dtype=np.float32))
                nutrients = np.asarray(nutrients, dtype=np.float32).flatten()
                meal_calories = nutrients[0]*4 + nutrients[1]*9 + nutrients[2]*9 + nutrients[3]*4 + nutrients[4]*2
                for i, key in enumerate(nutrient_keys):
                    self.state[f"daily_{key}_intake"] += float(nutrients[i])

                self.state["daily_calories_intake"] += meal_calories
                reward += self._calculate_reward(action_type, nutrients)

            elif action_type == 1:

                exercise_level = action.get("exercise_level", 0)
                met = [2.0, 6.0, 8.0][exercise_level]
                self.state["daily_calories_burned"] += (met * self.state["current_weight_kg"] * 3.5 / 200) * 60

            elif action_type == 2:

                rest_choice = action.get("rest_choice", 0)
                self.state["stress_level"] = max(self.min_stress_level, self.state["stress_level"] - [10,16][rest_choice])

        self.state["weight_history"].append(self.state["current_weight_kg"])
        self.state["stress_history"].append(self.state["stress_level"])

        self.state["time_of_day_slot"] += 1

        terminated = False
        if self.state["time_of_day_slot"] >= self.slots_per_day:

            self.state["time_of_day_slot"] = 0
            self.state["day_of_episode"] += 1
            net_calories = self.state["daily_calories_intake"] - self.state["daily_calories_burned"]
            self.state["current_weight_kg"] += net_calories / 7700

            daily_bmi_reward = 1 - abs(self._calculate_bmi() - self.target_bmi) / max(self.target_bmi, 0.0001)
            daily_nutrient_reward = sum(1 - abs(self.state[f"daily_{key}_intake"] - self.daily_targets_g[key]) / max(self.daily_targets_g[key], 0.0001) for key in nutrient_keys)

            for key in [f"daily_{k}_intake" for k in nutrient_keys] + ["daily_calories_intake", "daily_calories_burned"]:
                self.state[key] = 0.0

            reward += daily_bmi_reward + daily_nutrient_reward
            terminated = self.state["day_of_episode"] >= self.days_per_episode

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _calculate_reward(self, action_type: int, nutrients: np.ndarray = None) -> float:

        reward = 0.0
        if action_type == 0 and nutrients is not None:

            for i, key in enumerate(["protein", "fat", "saturated_fat", "carbs", "fiber"]):
                reward += 1 - abs(nutrients[i] - self.nutrients_target_per_meal[key]) / max(self.nutrients_target_per_meal[key], 0.0001)

        # Stress reward calculation
        ideal_stress = (self.min_stress_level + self.max_stress_level) / 4
        reward += 1 - abs(self.state["stress_level"] - ideal_stress) / max(self.max_stress_level - self.min_stress_level, 0.0001)
        return reward
