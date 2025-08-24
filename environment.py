import numpy as np
import gymnasium as gym
from typing import Optional
import copy

class LifeStyleCoachEnv(gym.Env):

    def __init__(self, initial_weight_kg: float = 65, height_cm: float = 170, age: int = 22,
                 gender: int = 0, target_bmi: float = 20, stress_range: tuple[float, float] = (0.0, 100.0),
                 days_per_episode: int = 28, work_mets: float = 2.0):
        

        super().__init__()


        # Static Variables
        self.initial_weight_kg = initial_weight_kg
        self.initial_height_cm = height_cm
        self.target_bmi = target_bmi
        self.initial_age = age
        self.initial_gender = gender    # Gender is male when 0, female when 1
        self.min_stress_level, self.max_stress_level = stress_range
        self.work_mets = work_mets

        self.state = {  # An object to store dynamic variables that changes during run time
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
            "stress_level_history" : [],
            "current_timeslot" : 0,
            "day_of_episode": 0,
            "time_since_last_meal": 0,
            "time_since_last_exercise": 0,
            "daily_calories_needed": initial_weight_kg * (30 if gender == 0 else 28)
        }

        self.daily_nutrients_target = {
            "protein": 0.18,        # Percentage
            "fat": 0.26,            # Percentage
            "saturated_fat": 0.05,  # Percentage
            "carbs": 0.51,          # Percentage
            "fiber": 30             # Grams 
        }

        self.update_daily_nutrient_targets()
        self.initial_state = copy.deepcopy(self.state)  # Make a copy of initial state to load in reset function
        
        self.slots_per_day = 24     # Each slot represents 1 hour in a day, so in total 24 hours a day
        self.days_per_episode = days_per_episode
        self.daily_schedule = ["action"] * self.slots_per_day

        # Setup schedule
        for start, end in [(10, 12), (13, 18)]:
            for i in range(start, end):
                self.daily_schedule[i] = 'work'


        for i in range(6):
            self.daily_schedule[i] = 'sleep'
        for i in range(22, 24):
            self.daily_schedule[i] = 'sleep'


        self.observation_space = gym.spaces.Dict(
            {
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
                "time_since_last_exercise": gym.spaces.Box(low=0, high=24, shape=(1,), dtype=np.int32),
                "daily_calories_needed": gym.spaces.Box(low=0, high=4000, shape=(1,), dtype=np.float32),
            }
        )


         
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )


    def rescale_action(self, action):
        lows  = np.array([0.0,   0,  0,  0,  0,  0,  2.0, 0.0], dtype=np.float32)
        highs = np.array([2.0, 120, 60,  8, 50, 20, 11.5, 1.0], dtype=np.float32)

        scaled = lows + (0.5 * (action + 1.0) * (highs - lows))
        return scaled

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

    def calculate_end_of_day_reward(self, prev_bmi):
        nutrients_reward_sum = 0
        reward = 0
        bmi_weight = 4
        nutrients_weight = 3
        calories_weight = 2

        # BMI Calculation first
        prev_distance_bmi = abs(prev_bmi - self.target_bmi)
        current_distance_bmi = abs(self.state["current_bmi"] - self.target_bmi)
        reward += (prev_distance_bmi - current_distance_bmi) * bmi_weight

        # Nutrient Calculation second 
        nutrients = ["protein", "fat", "saturated_fat", "carbs", "fiber"]
        for nutrient in nutrients:
            consumed = self.state[f"current_{nutrient}_intake"]
            target = self.daily_nutrients_target_g[nutrient]
            if target > 0:
                error = abs(consumed - target)
                normalized_error = error / target
            else:
                normalized_error = 0

            nutrients_reward_sum += max(0, 1 - normalized_error)

        nutrient_reward = (nutrients_reward_sum / len(nutrients))
        reward += nutrient_reward * nutrients_weight

        # Calories calculation third
        current_bmi = self.state["current_bmi"]

        if current_bmi < self.target_bmi - 0.5: 
            calorie_modifier = 500 
        elif current_bmi > self.target_bmi + 0.5: 
            calorie_modifier = -500 
        else: 
            calorie_modifier = 0 

        target_intake = self.state["daily_calories_needed"] + calorie_modifier
        actual_intake = self.state["current_calories_intake"]

        reward += np.exp(-0.5 * ((actual_intake - target_intake) / 200)**2) * calories_weight

        return reward

    def calculate_hourly_reward(self):
   
        reward = 1 / (1 + np.exp(0.1 * (self.state["current_stress_level"] - 40)))
            
        return (reward * 5)

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

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        action = self.rescale_action(action)

        main_choice = int(np.round(action[0]))
        nutrients = action[1:6]
        mets_level = action[6]
        rest_level = int(np.round(action[7]))


        terminated = False
        truncated = False
        reward = 0

        current_hour = self.state["current_timeslot"]
        event = self.daily_schedule[current_hour]
        
        if event == "work":
        
            self.state["current_stress_level"] = min(self.max_stress_level, self.state["current_stress_level"] + 5)
            self.state["current_calories_burned"] += (self.work_mets * self.state["current_weight_kg"] * 3.5 / 200) * 60

        elif event == "sleep":

            self.state["current_stress_level"] = max(self.min_stress_level, self.state["current_stress_level"] - 2)
            self.state["current_calories_burned"] += (0.9 * self.state["current_weight_kg"] * 3.5 / 200) * 60

        elif event == "action":
            
            if main_choice == 0:
                meal_calories = nutrients[0]*4 + nutrients[1]*9 + nutrients[2]*9 + nutrients[3]*4 + nutrients[4]*2

                self.state["current_calories_intake"] += meal_calories
                self.state["current_protein_intake"] += nutrients[0]
                self.state["current_fat_intake"] += nutrients[1]
                self.state["current_saturated_fat_intake"] += nutrients[2]
                self.state["current_carbs_intake"] += nutrients[3]
                self.state["current_fiber_intake"] += nutrients[4]

                self.state["time_since_last_meal"] = 0

            elif main_choice == 1:
                self.state["current_calories_burned"] += (mets_level * self.state["current_weight_kg"] * 3.5 / 200) * 60
                self.state["time_since_last_exercise"] = 0

            else:

                if rest_level == 0: 
                    rest_mets = 1.3 
                else: 
                    rest_mets = 1.8

                self.state["current_calories_burned"] += (rest_mets * self.state["current_weight_kg"] * 3.5 / 200) * 60
                self.state["current_stress_level"] = max(self.min_stress_level, self.state["current_stress_level"] - [10,16][rest_level])
        
        reward += self.calculate_hourly_reward()

        self.state["current_timeslot"] += 1

        self.state["time_since_last_meal"] = min(self.state["time_since_last_meal"] + 1, 24)
        self.state["time_since_last_exercise"] = min(self.state["time_since_last_exercise"] + 1, 24)

        self.state["bmi_history"].append(self.state["current_bmi"])
        self.state["stress_level_history"].append(self.state["current_stress_level"])

        if self.state["time_since_last_meal"] > 6:   
            reward -= 1 * (self.state["time_since_last_meal"] - 6)  

        if self.state["time_since_last_exercise"] > 48:  
            reward -= 2 * ((self.state["time_since_last_exercise"] - 48) // 24)  


        if self.state["current_timeslot"] >= self.slots_per_day:
            self.state["current_timeslot"] = 0

            calorie_balance = self.state["current_calories_intake"] - self.state["current_calories_burned"]

            prev_bmi_for_reward = self.state["current_bmi"]

            self.state["current_weight_kg"] += calorie_balance / 7700
            self.state["current_bmi"] = self.calculate_bmi()

            self.state["daily_calories_needed"] = self.calculate_daily_calories_needed()
            self.update_daily_nutrient_targets()

            self.state["day_of_episode"] += 1

            if abs(self.state["current_bmi"] - self.target_bmi) < 0.1:
                terminated = True
                reward += 100
            else:
                reward -= 0.1

            reward += self.calculate_end_of_day_reward(prev_bmi_for_reward) # Calcualte reward at the end of the day based on current state

            self.state["current_calories_burned"] = 0.0
            self.state["current_calories_intake"] = 0.0
            self.state["current_protein_intake"] = 0.0
            self.state["current_fat_intake"] = 0.0
            self.state["current_saturated_fat_intake"] = 0.0
            self.state["current_carbs_intake"] = 0.0
            self.state["current_fiber_intake"] = 0.0
            

        if self.state["current_bmi"] > 40 or self.state["current_bmi"] < 16:
            terminated = True
            reward -= 100

        truncated = self.state["day_of_episode"] >= self.days_per_episode

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info