import numpy as np
import gymnasium as gym
from typing import Optional
import copy

class LifeStyleCoachEnv(gym.Env):
    
    def __init__(self, initial_weight_kg: float = 65, height_cm: float = 170, age: int = 22,
                gender: int = 0, target_bmi: float = 20, stress_range: tuple[float, float] = (0.0, 100.0),
                days_per_episode: int = 28, work_mets: float = 2.0, exercise_target_ratio: float = 0.2):

        super().__init__()

        # Static Variables
        self.initial_weight_kg = initial_weight_kg
        self.initial_height_cm = height_cm
        self.target_bmi = target_bmi
        self.initial_age = age
        self.initial_gender = gender  # 0 = male, 1 = female
        self.min_stress_level, self.max_stress_level = stress_range
        self.work_mets = work_mets
        self.exercise_target_ratio = exercise_target_ratio

        self.state = {
            "current_weight_kg": initial_weight_kg,
            "current_bmi": initial_weight_kg / (height_cm / 100) ** 2,
            "current_stress_level": (self.min_stress_level + self.max_stress_level) / 2,
            "current_calories_burned": 0.0,
            "current_calories_intake": 0.0,
            "current_protein_intake": 0.0,
            "current_fat_intake": 0.0,
            "current_saturated_fat_intake": 0.0,
            "current_carbs_intake": 0.0,
            "current_fiber_intake": 0.0,
            "bmi_history": [],
            "stress_level_history": [],
            "current_timeslot": 0,
            "day_of_episode": 0,
            "time_since_last_meal": 0,
            "time_since_last_exercise": 0,
            "daily_calories_needed": initial_weight_kg * (30 if gender == 0 else 28)
        }

        self.daily_nutrients_target = {
            "protein": 0.18,
            "fat": 0.26,
            "saturated_fat": 0.05,
            "carbs": 0.51,
            "fiber": 30
        }

        self.update_daily_nutrient_targets()
        self.initial_state = copy.deepcopy(self.state)

        self.slots_per_day = 24
        self.days_per_episode = days_per_episode
        self.daily_schedule = ["action"] * self.slots_per_day

        # Setup work and sleep schedule
        for start, end in [(10, 12), (13, 18)]:
            for i in range(start, end):
                self.daily_schedule[i] = 'work'
        for i in range(6):
            self.daily_schedule[i] = 'sleep'
        for i in range(22, 24):
            self.daily_schedule[i] = 'sleep'

        self.action_space = gym.spaces.Discrete(8)

        self.observation_space = gym.spaces.Dict({
            "current_timeslot": gym.spaces.Discrete(24),
            "current_bmi": gym.spaces.Box(low=10.0, high=80.0, shape=(1,), dtype=np.float32),
            "current_calories_burned": gym.spaces.Box(low=0.0, high=6000.0, shape=(1,), dtype=np.float32),
            "current_calories_intake": gym.spaces.Box(low=0.0, high=6000.0, shape=(1,), dtype=np.float32),
            "current_protein_intake": gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32),
            "current_fat_intake": gym.spaces.Box(low=0.0, high=150.0, shape=(1,), dtype=np.float32),
            "current_saturated_fat_intake": gym.spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32),
            "current_carbs_intake": gym.spaces.Box(low=0.0, high=700.0, shape=(1,), dtype=np.float32),
            "current_fiber_intake": gym.spaces.Box(low=0.0, high=60.0, shape=(1,), dtype=np.float32),
            "current_weight_kg": gym.spaces.Box(low=40.0, high=300.0, shape=(1,), dtype=np.float32),
            "current_stress_level": gym.spaces.Box(low=self.min_stress_level, high=self.max_stress_level, shape=(1,), dtype=np.float32),
            "day_of_episode": gym.spaces.Box(low=0, high=self.days_per_episode, shape=(1,), dtype=np.int32),
            "time_since_last_meal": gym.spaces.Box(low=0, high=24, shape=(1,), dtype=np.int32),
            "time_since_last_exercise": gym.spaces.Box(low=0, high=48, shape=(1,), dtype=np.int32),
            "daily_calories_needed": gym.spaces.Box(low=0, high=4000, shape=(1,), dtype=np.float32),
        })

    def calculate_bmi(self):
        return self.state["current_weight_kg"] / (self.initial_height_cm / 100 ) ** 2
    
    def calculate_daily_calories_needed(self):
        return (self.state["current_weight_kg"] * (30 if self.initial_gender == 0 else 28))
    
    def update_daily_nutrient_targets(self):
        self.daily_nutrients_target_g = {
            "protein": (self.daily_nutrients_target["protein"] * self.state["daily_calories_needed"]) / 4, 
            "fat": (self.daily_nutrients_target["fat"] * self.state["daily_calories_needed"]) / 9, 
            "saturated_fat": (self.daily_nutrients_target["saturated_fat"] * self.state["daily_calories_needed"]) / 9, 
            "carbs": (self.daily_nutrients_target["carbs"] * self.state["daily_calories_needed"]) / 4, 
            "fiber": self.daily_nutrients_target["fiber"] 
        }

    def _get_obs(self):
        return {
            "current_timeslot": self.state["current_timeslot"],
            "current_bmi": np.array([self.state["current_bmi"]], dtype=np.float32),
            "current_calories_burned": np.array([self.state["current_calories_burned"]], dtype=np.float32),
            "current_calories_intake": np.array([self.state["current_calories_intake"]], dtype=np.float32),
            "current_protein_intake": np.array([self.state["current_protein_intake"]], dtype=np.float32),
            "current_fat_intake": np.array([self.state["current_fat_intake"]], dtype=np.float32),
            "current_saturated_fat_intake": np.array([self.state["current_saturated_fat_intake"]], dtype=np.float32),
            "current_carbs_intake": np.array([self.state["current_carbs_intake"]], dtype=np.float32),
            "current_fiber_intake": np.array([self.state["current_fiber_intake"]], dtype=np.float32),
            "current_weight_kg": np.array([self.state["current_weight_kg"]], dtype=np.float32),
            "current_stress_level": np.array([self.state["current_stress_level"]], dtype=np.float32),
            "day_of_episode": np.array([self.state["day_of_episode"]], dtype=np.int32),
            "time_since_last_meal": np.array([self.state["time_since_last_meal"]], dtype=np.int32),
            "time_since_last_exercise": np.array([self.state["time_since_last_exercise"]], dtype=np.int32),
            "daily_calories_needed": np.array([self.state["daily_calories_needed"]], dtype=np.float32),
        }
        
    def _get_info(self):
        return {
            "bmi_progress": abs(self.state["current_bmi"] - self.target_bmi),
            "weight_progress": self.initial_weight_kg - self.state["current_weight_kg"],
            "calorie_balance": self.state["current_calories_intake"] - self.state["current_calories_burned"],
            "stress_level_history": list(self.state["stress_level_history"]),
            "bmi_history": list(self.state["bmi_history"]),
            "days_remaining": self.days_per_episode - self.state["day_of_episode"],
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)

        self.state = copy.deepcopy(self.initial_state)

        self.state["bmi_history"] = []
        self.state["stress_level_history"] = []
        self.state["time_since_last_meal"] = 0
        self.state["time_since_last_exercise"] = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def get_action_mask(self):
        current_hour = self.state["current_timeslot"]
        fixed_event = self.daily_schedule[current_hour]

        mask = np.ones(self.action_space.n, dtype=np.int32) 
        
        if fixed_event in ["work", "sleep"]:
            mask[:] = 0  
        return mask


    def step(self, action):

        terminated = False
        truncated = False
        reward = 0

        current_hour = self.state["current_timeslot"]
        event = self.daily_schedule[current_hour]

        if event == "work":
            self.state["current_stress_level"] = min(self.max_stress_level, self.state["current_stress_level"] + 4)
            self.state["current_calories_burned"] += (self.work_mets * self.state["current_weight_kg"] * 3.5 / 200) * 60

        elif event == "sleep":
            self.state["current_stress_level"] = max(self.min_stress_level, self.state["current_stress_level"] - 2)
            self.state["current_calories_burned"] += (0.9 * self.state["current_weight_kg"] * 3.5 / 200) * 60

        else:

            if action in [0, 1, 2]:  # Meal
                meal_calories = [350, 775, 1200][action]  

                self.state["current_calories_intake"] += meal_calories
                self.state["time_since_last_meal"] = 0

                self.state["current_protein_intake"] += (self.daily_nutrients_target["protein"] * meal_calories) / 4
                self.state["current_fat_intake"] += (self.daily_nutrients_target["fat"] * meal_calories) / 9
                self.state["current_saturated_fat_intake"] += (self.daily_nutrients_target["saturated_fat"] * meal_calories) / 9
                self.state["current_carbs_intake"] += (self.daily_nutrients_target["carbs"] * meal_calories) / 4
                self.state["current_fiber_intake"] += self.daily_nutrients_target["fiber"]  

                reward += self.calculate_meal_reward(meal_calories)

            elif action in [3, 4, 5]:  # Exercise
                mets_levels = [2.0, 4.5, 9.0]  
                calories_burned = (mets_levels[action-3] * self.state["current_weight_kg"] * 3.5 / 200) * 60
                self.state["current_calories_burned"] += calories_burned
                self.state["time_since_last_exercise"] = 0

                reward += self.calculate_exercise_reward(calories_burned)

            elif action in [6, 7]:  # Rest
                rest_levels = [1.3, 1.8]  
                self.state["current_calories_burned"] += (rest_levels[action-6] * self.state["current_weight_kg"] * 3.5 / 200) * 60
                self.state["current_stress_level"] = max(self.min_stress_level, self.state["current_stress_level"] - [10,16][action-6])

            reward += self.calculate_stress_rewards()
            reward += self.calculate_last_meal_exercise(action)

        self.state["current_timeslot"] += 1

        self.state["bmi_history"].append(self.state["current_bmi"])
        self.state["stress_level_history"].append(self.state["current_stress_level"])

        if self.state["current_timeslot"] >= self.slots_per_day:
            self.state["current_timeslot"] = 0

            calorie_balance = self.state["current_calories_intake"] - self.state["current_calories_burned"]
            prev_bmi_for_reward = self.state["current_bmi"]

            self.state["current_weight_kg"] += (calorie_balance / 7700)
            self.state["current_bmi"] = self.calculate_bmi()

            self.state["daily_calories_needed"] = self.calculate_daily_calories_needed()
            self.update_daily_nutrient_targets()

            self.state["day_of_episode"] += 1

            reward += self.calculate_BMI_reward(prev_bmi_for_reward)

            if abs(self.state["current_bmi"] - self.target_bmi) < 0.2:
                terminated = True
                reward += 5

            reset_keys = [
                "current_calories_burned",
                "current_calories_intake",
                "current_protein_intake",
                "current_fat_intake",
                "current_saturated_fat_intake",
                "current_carbs_intake",
                "current_fiber_intake"
            ]
            
            for key in reset_keys:
                self.state[key] = 0.0

        if self.state["current_bmi"] > 40 or self.state["current_bmi"] < 16:
            terminated = True       
            reward -= 5

        observation = self._get_obs()
        info = self._get_info()
        truncated = self.state["day_of_episode"] >= self.days_per_episode

        return observation, reward, terminated, truncated, info
    
    def calculate_meal_reward(self, meal_calories):
        nutrients_weight = 1

        meal_nutrients = {
            "protein": (self.daily_nutrients_target["protein"] * meal_calories) / 4,
            "fat": (self.daily_nutrients_target["fat"] * meal_calories) / 9,
            "saturated_fat": (self.daily_nutrients_target["saturated_fat"] * meal_calories) / 9,
            "carbs": (self.daily_nutrients_target["carbs"] * meal_calories) / 4,
            "fiber": self.daily_nutrients_target["fiber"]  
        }

        nutrients_reward_sum = 0
        nutrients = ["protein", "fat", "saturated_fat", "carbs", "fiber"]

        for nutrient in nutrients:
            consumed = meal_nutrients[nutrient]
            target = self.daily_nutrients_target_g[nutrient]
            if target > 0:
                error = abs(consumed - target)
                normalized_error = error / target
            else:
                normalized_error = 0

            nutrients_reward_sum += max(0, 1 - normalized_error)

        nutrient_reward = (nutrients_reward_sum / len(nutrients)) * nutrients_weight
        return nutrient_reward

    def calculate_exercise_reward(self, calories_burned):
        exercise_weight = 1
        target_calories_burned = self.exercise_target_ratio * self.state["daily_calories_needed"]
        normalized_reward = min(calories_burned / target_calories_burned, 1.0)
        reward = exercise_weight * normalized_reward
        return reward

    def calculate_stress_rewards(self):
        stress_min, stress_max = self.min_stress_level, self.max_stress_level
        current_stress = self.state["current_stress_level"]
        
        reward = 1.0 - (current_stress - stress_min) / (stress_max - stress_min)
        reward = np.clip(reward, 0.0, 1.0)

        consecutive_high = sum(1 for s in self.state["stress_level_history"][-10:] if s > 80)
        reward -= 0.05 * consecutive_high  # small penalty
        reward = np.clip(reward, 0.0, 1.0)

        return reward

    
    def calculate_BMI_reward(self, prev_bmi):
        prev_distance = abs(prev_bmi - self.target_bmi)
        current_distance = abs(self.state["current_bmi"] - self.target_bmi)
        improvement = prev_distance - current_distance
        return np.clip(improvement * 10, -1.0, 1.0)  # scale to [-1,1]
    
    def calculate_last_meal_exercise(self, action):
        self.state["time_since_last_meal"] = min(self.state["time_since_last_meal"] + 1, 24) if action not in [0,1,2] else 0
        self.state["time_since_last_exercise"] = min(self.state["time_since_last_exercise"] + 1, 48) if action not in [3,4,5] else 0

        meal_penalty = self.state["time_since_last_meal"] / 24   
        exercise_penalty = self.state["time_since_last_exercise"] / 48  

        penalty = -0.5 * meal_penalty - 0.5 * exercise_penalty
        penalty = np.clip(penalty, -0.5, 0.0)  

        return penalty


