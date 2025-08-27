import numpy as np
import gymnasium as gym
from typing import Optional
import copy

class LifeStyleEnv(gym.Env):

    def __init__(self, initial_weight_kg: float = 70, height_cm: float = 170, gender: int = 0,
                 target_bmi: float = 21.75, stress_range: tuple[float, float] = (0.0, 100.0),
                 hunger_range: tuple[float, float] = (0.0, 100.0), energy_range: tuple[float, float] = (0.0, 100.0),
                 days_per_episode: int = 96, work_mets: float = 2.0):

        super().__init__()

        # Static Variables 
        self.initial_weight_kg = initial_weight_kg
        self.height_cm = height_cm
        self.gender = gender
        self.target_bmi = target_bmi
        self.min_stress, self.max_stress = stress_range
        self.min_hunger, self.max_hunger = hunger_range
        self.min_energy, self.max_energy = energy_range
        self.days_per_episode = days_per_episode
        self.work_mets = work_mets

        # Define Schedule
        self.slots_per_day = 24
        self.daily_schedule = ["action"] * self.slots_per_day

        # Setup work and sleep schedule
        # 10:00 AM to 12:00 PM and 1:00 PM to 6:00 PM
        for start, end in [(10, 12), (13, 18)]:
            for i in range(start, end):
                self.daily_schedule[i] = 'work'

        # Sleep: 10:00 PM to 6:00 AM
        for i in range(6):
            self.daily_schedule[i] = 'sleep' 
        for i in range(22, 24):
            self.daily_schedule[i] = 'sleep'

        self.action_space = gym.spaces.Discrete(9)

        self.daily_nutrients_target = {
            "protein": 0.18,
            "fat": 0.26,
            "saturated_fat": 0.05,
            "carbs": 0.51,
            "fiber": 30
        }

        self.state = {
            "current_weight_kg": self.initial_weight_kg,
            "current_bmi": self.initial_weight_kg / (self.height_cm / 100) ** 2,
            "current_stress_level": (self.min_stress + self.max_stress) / 2,
            "current_hunger_level": (self.min_hunger + self.max_hunger) / 2,
            "current_energy_level": (self.min_energy + self.max_energy) / 2,
            
            "daily_calories_burned": 0.0,
            "daily_calories_intake": 0.0,
            "daily_protein_intake": 0.0,
            "daily_fat_intake": 0.0,
            "daily_saturated_fat_intake": 0.0,
            "daily_carbs_intake": 0.0,
            "daily_fiber_intake": 0.0,
            "daily_calories_needed": self.initial_weight_kg * (30 if self.gender == 0 else 28),

            "current_timeslot": 0,
            "day_of_episode": 0,
            "time_since_last_meal": 0,
            "time_since_last_exercise": 0,
            "bmi_history": [],
            "stress_level_history": []
        }

        self.observation_space = gym.spaces.Dict({
            "current_weight_kg": gym.spaces.Box(low=30.0, high=300.0, shape=(1,), dtype=np.float32), 
            "current_bmi": gym.spaces.Box(low=10.0, high=80.0, shape=(1,), dtype=np.float32),     
            "current_stress_level": gym.spaces.Box(low=self.min_stress, high=self.max_stress, shape=(1,), dtype=np.float32),
            "current_hunger_level": gym.spaces.Box(low=self.min_hunger, high=self.max_hunger, shape=(1,), dtype=np.float32),
            "current_energy_level": gym.spaces.Box(low=self.min_energy, high=self.max_energy, shape=(1,), dtype=np.float32),
            
            "daily_calories_burned": gym.spaces.Box(low=0.0, high=6000.0, shape=(1,), dtype=np.float32), 
            "daily_calories_intake": gym.spaces.Box(low=0.0, high=6000.0, shape=(1,), dtype=np.float32), 
            "daily_protein_intake": gym.spaces.Box(low=0.0, high=600.0, shape=(1,), dtype=np.float32),   
            "daily_fat_intake": gym.spaces.Box(low=0.0, high=400.0, shape=(1,), dtype=np.float32),      
            "daily_saturated_fat_intake": gym.spaces.Box(low=0.0, high=200.0, shape=(1,), dtype=np.float32), 
            "daily_carbs_intake": gym.spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32),   
            "daily_fiber_intake": gym.spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),    
            "daily_calories_needed": gym.spaces.Box(low=0.0, high=4000.0, shape=(1,), dtype=np.float32), 

            "current_timeslot": gym.spaces.Discrete(self.slots_per_day), 
            "day_of_episode": gym.spaces.Box(low=0, high=self.days_per_episode, shape=(1,), dtype=np.int32),
            "time_since_last_meal": gym.spaces.Box(low=0, high=self.slots_per_day, shape=(1,), dtype=np.int32),
            "time_since_last_exercise": gym.spaces.Box(low=0, high=self.days_per_episode * self.slots_per_day, shape=(1,), dtype=np.int32)
        })

        self.update_daily_nutrient_targets()    

        self.initial_state = copy.deepcopy(self.state)

    def _get_obs(self):

        obs = {
            "current_weight_kg": np.array([self.state["current_weight_kg"]], dtype=np.float32),
            "current_bmi": np.array([self.state["current_bmi"]], dtype=np.float32),
            "current_stress_level": np.array([self.state["current_stress_level"]], dtype=np.float32),
            "current_hunger_level": np.array([self.state["current_hunger_level"]], dtype=np.float32),
            "current_energy_level": np.array([self.state["current_energy_level"]], dtype=np.float32),
            
            "daily_calories_burned": np.array([self.state["daily_calories_burned"]], dtype=np.float32),
            "daily_calories_intake": np.array([self.state["daily_calories_intake"]], dtype=np.float32),
            "daily_protein_intake": np.array([self.state["daily_protein_intake"]], dtype=np.float32),
            "daily_fat_intake": np.array([self.state["daily_fat_intake"]], dtype=np.float32),
            "daily_saturated_fat_intake": np.array([self.state["daily_saturated_fat_intake"]], dtype=np.float32),
            "daily_carbs_intake": np.array([self.state["daily_carbs_intake"]], dtype=np.float32),
            "daily_fiber_intake": np.array([self.state["daily_fiber_intake"]], dtype=np.float32),
            "daily_calories_needed": np.array([self.state["daily_calories_needed"]], dtype=np.float32),
            
            "current_timeslot": self.state["current_timeslot"],
            "day_of_episode": np.array([self.state["day_of_episode"]], dtype=np.int32),
            "time_since_last_meal": np.array([self.state["time_since_last_meal"]], dtype=np.int32),
            "time_since_last_exercise": np.array([self.state["time_since_last_exercise"]], dtype=np.int32)
        }
        
        return obs

    def _get_info(self):
        return {
            "bmi_progress": abs(self.state["current_bmi"] - self.target_bmi),
            "weight_progress": self.initial_weight_kg - self.state["current_weight_kg"],
            "calorie_balance": self.state["daily_calories_intake"] - self.state["daily_calories_burned"],
            "stress_level_history": list(self.state["stress_level_history"]),
            "bmi_history": list(self.state["bmi_history"]),
            "days_remaining": self.days_per_episode - self.state["day_of_episode"],
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        self.state = copy.deepcopy(self.initial_state)
        observation = self._get_obs()
        info = self._get_info()

        mask = self.get_action_mask()
        info["action_mask"] = mask 

        return observation, info

    def get_action_mask(self):
        current_event = self.daily_schedule[self.state["current_timeslot"]]
        current_energy = self.state["current_energy_level"]

        mask = [True] * self.action_space.n
        skip_action_index = 8
        intense_exercise_index = 5
        moderate_exercise_index = 4
        light_exercise_index = 3

        if current_event in ['sleep', 'work']:
            mask = [False] * self.action_space.n
            mask[skip_action_index] = True
        else:
            mask[skip_action_index] = False
            if current_energy < 20:
                mask[intense_exercise_index] = False
            if current_energy < 15:
                mask[moderate_exercise_index] = False
            if current_energy < 10:
                mask[light_exercise_index] = False

        return np.array(mask, dtype=bool)

    def step(self, action):

        reward = 0.0
        terminated = False
        truncated = False
        
        current_timeslot = self.state["current_timeslot"]
        schedule_event = self.daily_schedule[current_timeslot]

        if schedule_event == "work":
            self.state["current_stress_level"] = min(self.max_stress, self.state["current_stress_level"] + 5)
            self.state["current_energy_level"] = max(self.min_energy, self.state["current_energy_level"] - 5)
            self.state["current_hunger_level"] = min(self.max_hunger, self.state["current_hunger_level"] + 5)
            self.state["daily_calories_burned"] += (self.work_mets * self.state["current_weight_kg"] * 3.5 / 200) * 60
        
        elif schedule_event == "sleep":
            self.state["current_energy_level"] = min(self.max_energy, self.state["current_energy_level"] + 4)
            self.state["current_stress_level"] = max(self.min_stress, self.state["current_stress_level"] - 4)
            self.state["current_hunger_level"] = max(self.min_hunger, self.state["current_hunger_level"] - 2)
            self.state["daily_calories_burned"] += (0.9 * self.state["current_weight_kg"] * 3.5 / 200) * 60

        else:
            if action in [0, 1, 2]:

                meal_data = [
                    {"calories": 350, "fiber_g": 5, "protein_ratio": 0.20, "fat_ratio": 0.25, "sat_fat_ratio": 0.05, "carbs_ratio": 0.55}, # Light meal
                    {"calories": 775, "fiber_g": 10, "protein_ratio": 0.18, "fat_ratio": 0.26, "sat_fat_ratio": 0.06, "carbs_ratio": 0.56}, # Medium meal
                    {"calories": 1200, "fiber_g": 15, "protein_ratio": 0.15, "fat_ratio": 0.30, "sat_fat_ratio": 0.07, "carbs_ratio": 0.55}, # Heavy meal
                ]

                selected_meal = meal_data[action]
                
                self.state["daily_calories_intake"] += selected_meal["calories"]
                self.state["time_since_last_meal"] = 0
                
                self.state["daily_protein_intake"] += (selected_meal["protein_ratio"] * selected_meal["calories"]) / 4
                self.state["daily_fat_intake"] += (selected_meal["fat_ratio"] * selected_meal["calories"]) / 9
                self.state["daily_saturated_fat_intake"] += (selected_meal["sat_fat_ratio"] * selected_meal["calories"]) / 9
                self.state["daily_carbs_intake"] += (selected_meal["carbs_ratio"] * selected_meal["calories"]) / 4
                self.state["daily_fiber_intake"] += selected_meal["fiber_g"]

                self.state["current_hunger_level"] = max(self.min_hunger, self.state["current_hunger_level"] - [20,30,50][action])
                self.state["current_energy_level"] = min(self.max_energy, self.state["current_energy_level"] + [20,30,50][action])

            
            elif action in [3, 4, 5]:
                mets_levels = [2.0, 4.5, 9.0]  
                calories_burned = (mets_levels[action-3] * self.state["current_weight_kg"] * 3.5 / 200) * 60
                self.state["daily_calories_burned"] += calories_burned
                self.state["current_energy_level"] = max(self.min_energy, self.state["current_energy_level"] - [5,10,15][action-3])
                self.state["current_hunger_level"] = min(self.max_hunger, self.state["current_hunger_level"] + [5,10,15][action-3])
                self.state["time_since_last_exercise"] = 0

            elif action in [6, 7]:
                rest_levels = [1.2, 1.0]  
                self.state["daily_calories_burned"] += (rest_levels[action-6] * self.state["current_weight_kg"] * 3.5 / 200) * 60
                self.state["current_stress_level"] = max(self.min_stress, self.state["current_stress_level"] - [10,16][action-6])
                self.state["current_energy_level"] = min(self.max_energy, self.state["current_energy_level"] + [10,16][action-6])
                self.state["current_hunger_level"] = min(self.max_hunger, self.state["current_hunger_level"] + 2)
                
            elif action in [8]:
                pass


        reward += self.get_reward()

        if self.state["current_timeslot"] >= self.slots_per_day - 1:
            calorie_balance = self.state["daily_calories_intake"] - self.state["daily_calories_burned"]

            self.state["current_weight_kg"] += (calorie_balance / 7700)
            self.state["current_bmi"] = self.state["current_weight_kg"] / (self.height_cm / 100) ** 2
            
            self.state["daily_calories_needed"] = self.state["current_weight_kg"] * (30 if self.gender == 0 else 28)

            self.state["bmi_history"].append(self.state["current_bmi"])
            self.state["stress_level_history"].append(self.state["current_stress_level"])

            bmi = self.state["current_bmi"]
            target_bmi = self.target_bmi
            bmi_difference = abs(bmi - target_bmi)
            reward += 10.0 - (2 * (bmi_difference / max(target_bmi, 0.0001))) 

            if abs(self.state["current_bmi"] - self.target_bmi) < 0.2:
                terminated = True
                reward += 100

            self.update_daily_nutrient_targets()

            reset_keys = [
                "daily_calories_burned",
                "daily_calories_intake",
                "daily_protein_intake",
                "daily_fat_intake",
                "daily_saturated_fat_intake",
                "daily_carbs_intake",
                "daily_fiber_intake"
            ]
            
            for key in reset_keys:
                self.state[key] = 0.0

            self.state["current_timeslot"] = 0
            self.state["day_of_episode"] += 1
        
        else:
            self.state["current_timeslot"] += 1


        self.state["time_since_last_meal"] += 1
        self.state["time_since_last_exercise"] += 1

        if self.state["current_bmi"] > 40 or self.state["current_bmi"] < 16:
            terminated = True       
            reward -= 100

        observation = self._get_obs()
        info = self._get_info()
        truncated = self.state["day_of_episode"] >= self.days_per_episode 

        return observation, reward, terminated, truncated, info
    
    def update_daily_nutrient_targets(self):
        self.daily_nutrients_target_g = {
            "protein": (self.daily_nutrients_target["protein"] * self.state["daily_calories_needed"]) / 4, 
            "fat": (self.daily_nutrients_target["fat"] * self.state["daily_calories_needed"]) / 9, 
            "saturated_fat": (self.daily_nutrients_target["saturated_fat"] * self.state["daily_calories_needed"]) / 9, 
            "carbs": (self.daily_nutrients_target["carbs"] * self.state["daily_calories_needed"]) / 4, 
            "fiber": self.daily_nutrients_target["fiber"] 
        }

    def get_reward(self):
        reward = 0.0

        current_stress = self.state["current_stress_level"]
        stress_min, stress_max = self.min_stress, self.max_stress
        reward += 1.0 - (2 * (current_stress - stress_min) / (stress_max - stress_min))


        hunger_level = self.state["current_hunger_level"]
        normalized_hunger_above_threshold = max(0, hunger_level - 30) / (self.max_hunger - 30)
        hunger_penalty_factor = normalized_hunger_above_threshold ** 2.0
        hunger_penalty = hunger_penalty_factor * - 15
        reward += hunger_penalty 


        time_since_meal = self.state["time_since_last_meal"]
        time_above_threshold = max(0, time_since_meal - 5)
        meal_timing_penalty = max(-20.0, time_above_threshold * -2)
        reward += meal_timing_penalty


        nutrients_reward_sum = 0
        nutrient_list = ["protein", "fat", "carbs", "saturated_fat", "fiber"]
        for nutrient in nutrient_list:
            target = self.daily_nutrients_target_g[nutrient]
            consumed = self.state[f"daily_{nutrient}_intake"]
            if target > 0:
                error = abs(consumed - target)
                normalized_error = error / target
            else:
                normalized_error = 0

            nutrients_reward_sum += max(0, 2 - normalized_error)
        
        reward += (nutrients_reward_sum / 5)


        calories_reward = (self.state["daily_calories_burned"] - self.state["daily_calories_intake"]) * 0.0035
        reward += calories_reward

        return reward
    
    def action_masks(self) -> np.ndarray:
        return self.get_action_mask()