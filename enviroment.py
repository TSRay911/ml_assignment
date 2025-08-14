import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class LifeStyleCoachEnv(gym.Env):

    # Static parameters to initialize environment
    def __init__(self, inital_weight_kg: float = 80, height_cm: float = 165, age: int = 22,
                 gender: int = 0, target_bmi: float = 20, stress_range: tuple[float, float] = (1.0, 100.0),
                 energy_range: tuple[float, float] = (0.0, 100.0), days_per_episode: int = 28):
    
        super(LifeStyleCoachEnv, self).__init__()

        # Static variables
        self.initial_weight_kg = inital_weight_kg
        self.height_cm = height_cm
        self.age = age
        self.gender = gender
        self.target_bmi = target_bmi
        self.min_energy_range, self.max_energy_range = energy_range
        self.days_per_episode = days_per_episode
        self.min_stress_level, self.max_stress_level = stress_range

        self.daily_calories_goal = self.calculate_daily_calories()

        self.daily_targets = {
            "protein": 0.2,          # 20% of calories
            "fat": 0.3,              # 30% of calories
            "saturated_fat": 0.1,    # 10% of calories
            "carbs": 0.5,            # 50% of calories
            "fiber": 30              # g/day
        }

        self.daily_targets_g = {
            "protein": (self.daily_targets["protein"] * self.daily_calories_goal) / 4,
            "fat": (self.daily_targets["fat"] * self.daily_calories_goal) / 9,
            "saturated_fat": (self.daily_targets["saturated_fat"] * self.daily_calories_goal) / 9,
            "carbs": (self.daily_targets["carbs"] * self.daily_calories_goal) / 4,
            "fiber": self.daily_targets["fiber"]  
        }

        meals_per_day = 3
        self.meal_targets = {}

        for nutrient, total_grams in self.daily_targets_g.items():
            self.meal_targets[nutrient] = total_grams / meals_per_day


        # Dynamic variables
        self.state = {
            "current_weight_kg": self.initial_weight_kg,
            "time_of_day_slot": 0,
            "day_of_episode": 0,
            "daily_calories_intake": 0.0,
            "daily_calories_burned": 0.0,
            "daily_protein_intake": 0.0,
            "daily_fat_intake": 0.0,
            "daily_carbohydrate_intake": 0.0,
            "daily_saturated_fat_intake": 0.0,
            "daily_fiber_intake": 0.0,
            "stress_level": (self.min_stress_level + self.max_stress_level) / 2, 
            "weight_history": [],
            "stress_history": [],
            "energy_level": (self.min_energy_range + self.max_energy_range) / 2,
            "energy_history": []
        }

        self.slots_per_day = 24
        self.daily_schedule = ['agent_controlled_action'] * self.slots_per_day

        work_slots = [(10, 12), (13, 17)]

        # 10 am - 12 pm work, 12 pm to 1 pm break, 1 pm to 6 pm work
        for start_hour, end_hour in work_slots:
            for hour in range(start_hour, end_hour):
                self.daily_schedule[hour] = 'work'
        
        # 10 pm sleep and 6am wake up
        for hour in range(6): 
            self.daily_schedule[hour] = 'sleep'
        for hour in range(22, 24):
            self.daily_schedule[hour] = 'sleep'


        self.observation_space = gym.spaces.Dict(
            {
                "current_timeslot": gym.spaces.Discrete(24),
                "current_bmi": gym.spaces.Box(low=10.0, high=80.0, shape=(), dtype=np.float32),
                "daily_calories_intake": gym.spaces.Box(low=0.0, high=6000.0, shape=(), dtype=np.float32),
                "daily_calories_burned": gym.spaces.Box(low=0.0, high=6000.0, shape=(), dtype=np.float32),
                "daily_protein_intake": gym.spaces.Box(low=0.0, high=300.0, shape=(), dtype=np.float32),
                "daily_fat_intake": gym.spaces.Box(low=0.0, high=150.0, shape=(), dtype=np.float32),
                "daily_saturated_fat_intake": gym.spaces.Box(low=0.0, high=50.0, shape=(), dtype=np.float32),
                "daily_carbohydrate_intake": gym.spaces.Box(low=0.0, high=700.0, shape=(), dtype=np.float32),
                "daily_fiber_intake": gym.spaces.Box(low=0.0, high=60.0, shape=(), dtype=np.float32),
                "current_weight_kg": gym.spaces.Box(low=40.0, high=300.0, shape=(), dtype=np.float32),
                "stress_level": gym.spaces.Box(low=self.min_stress_level, high=self.max_stress_level, shape=(), dtype=np.float32),
                "day_of_episode": gym.spaces.Box(low=0, high=self.days_per_episode, shape=(), dtype=np.int32),
                "energy_level": gym.spaces.Box(low=self.min_energy_range, high=self.max_energy_range, shape=(), dtype=np.float32)
            }
        )

        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(3), 
            "action_details": spaces.MultiDiscrete([
                4, 4, 4, 4, 4,    # Meal choices
                3,                # Exercise choices
                2                 # Rest choices
            ])
        })


    def calculate_daily_calories(self):
        weight = self.state["current_weight_kg"]
        gender = self.gender  
        if gender == 0:
            return weight * 30  
        else:
            return weight * 28  



    def _calculate_bmi(self):
        
        if self.height_cm <= 0:
            return 0.0

        return self.state["current_weight_kg"] / ((self.height_cm / 100) ** 2)


    def _get_obs(self):

        current_bmi = self._calculate_bmi()

        return {
            "current_timeslot": self.state["time_of_day_slot"],
            "current_bmi": np.array(current_bmi, dtype=np.float32),
            "daily_calories_intake": np.array(self.state["daily_calories_intake"], dtype=np.float32),
            "daily_calories_burned": np.array(self.state["daily_calories_burned"], dtype=np.float32),
            "daily_protein_intake": np.array(self.state["daily_protein_intake"], dtype=np.float32),
            "daily_fat_intake": np.array(self.state["daily_fat_intake"], dtype=np.float32),
            "daily_saturated_fat_intake": np.array(self.state["daily_saturated_fat_intake"], dtype=np.float32),
            "daily_carbohydrate_intake": np.array(self.state["daily_carbohydrate_intake"], dtype=np.float32),
            "daily_fiber_intake": np.array(self.state["daily_fiber_intake"], dtype=np.float32),
            "current_weight_kg": np.array(self.state["current_weight_kg"], dtype=np.float32),
            "stress_level": np.array(self.state["stress_level"], dtype=np.float32), 
            "day_of_episode": np.array(self.state["day_of_episode"], dtype=np.int32),
            "energy_level": np.array(self.state["energy_level"], dtype=np.float32)
        }
    
    
    def _get_info(self):
        return {
            "bmi_progress": self.target_bmi - self._calculate_bmi(),
            "weight_progress_kg": self.initial_weight_kg - self.state["current_weight_kg"],
            "calorie_balance": self.state["daily_calories_intake"] - self.state["daily_calories_burned"],
            "stress_history": list(self.state["stress_history"]),
            "weight_history": list(self.state["weight_history"]),
            "days_remaining": self.days_per_episode - self.state["day_of_episode"]
        }

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state.update({
            "current_weight_kg": self.initial_weight_kg,
            "time_of_day_slot": 0,
            "day_of_episode": 0,
            "daily_calories_intake": 0.0,
            "daily_calories_burned": 0.0,
            "daily_protein_intake": 0.0,
            "daily_fat_intake": 0.0,
            "daily_carbohydrate_intake": 0.0,
            "daily_saturated_fat_intake": 0.0,
            "daily_fiber_intake": 0.0,
            "stress_level": (self.min_stress_level + self.max_stress_level) / 2,
            "weight_history": [],
            "stress_history": [],
            "energy_level": (self.min_energy_range + self.max_energy_range) / 2,
            "energy_history": [],
        })
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        done = False
        current_hour = self.state["time_of_day_slot"]
        event = self.daily_schedule[current_hour]
        reward = 0.0

        if event == "sleep":
            self.state["stress_level"] = max(self.min_stress_level, self.state["stress_level"] - 5)
            self.state["energy_level"] = min(self.max_energy_level, self.state["energy_level"] + 10)
            weight = self.state["current_weight_kg"]
            sleep_met = 0.9  
            minutes_sleep = 8 * 60  
            calories_burned_sleep = (sleep_met * weight * 3.5 / 200) * minutes_sleep
            self.state["daily_calories_burned"] += calories_burned_sleep

        elif event == "work":
            # Work increases stress and burns calories
            self.state["stress_level"] = min(self.max_stress_level, self.state["stress_level"] + 4)
            self.state["energy_level"] = max(self.min_energy_level, self.state["energy_level"] - 6)
            weight = self.state["current_weight_kg"]
            work_met = 2.0  
            minutes_work = 60  
            calories_burned_work = (work_met * weight * 3.5 / 200) * minutes_work
            self.state["daily_calories_burned"] += calories_burned_work

        else:

            action_type = action["action_type"]
            details = action["action_details"]

            if action_type == 0:  # Eat

                carb_g = [0, 30, 50, 70]      
                protein_g = [0, 10, 20, 30]       
                fat_g = [0, 5, 10, 15]           
                sat_fat_g = [0, 2, 4, 6]          
                fiber_g = [0, 5, 10, 15]          

                meal_calories = (
                    carb_g[details[0]] * 4 +
                    protein_g[details[1]] * 4 +
                    fat_g[details[2]] * 9 +
                    sat_fat_g[details[3]] * 9 + 
                    fiber_g[details[4]] * 2
                )

                # Update daily intake
                self.state["daily_carbohydrate_intake"] += carb_g[details[0]]
                self.state["daily_protein_intake"] += protein_g[details[1]]
                self.state["daily_fat_intake"] += fat_g[details[2]]
                self.state["daily_saturated_fat_intake"] += sat_fat_g[details[3]]
                self.state["daily_fiber_intake"] += fiber_g[details[4]]
                self.state["daily_calories_intake"] += meal_calories


                # Eating restores energy
                energy_gain = meal_calories / 2000 * self.max_energy_level

                self.state["energy_level"] = min(self.max_energy_level,
                                 self.state["energy_level"] + energy_gain)
                
                reward += self._calculate_reward(action_type)


            elif action_type == 1:  # Exercise
                
                intensity = details[5]
                exercise_mets = [4.0, 6.0, 8.0]
                met = exercise_mets[intensity]
                
                weight = self.state["current_weight_kg"]
                minutes_exercise = 60  
                calories_burned_exercise = (met * weight * 3.5 / 200) * minutes_exercise
                
                self.state["daily_calories_burned"] += calories_burned_exercise
                
                fatigue_levels = [10, 20, 30]  
                self.state["energy_level"] = max(self.min_energy_level, self.state["energy_level"] - fatigue_levels[intensity])

                reward += self._calculate_reward(action_type)

            elif action_type == 2:  # Rest
                rest_levels = [5, 10]  # energy recovery
                rest_choice = details[6]
                self.state["energy_level"] = min(self.max_energy_level, self.state["energy_level"] + rest_levels[rest_choice])

                reward += self._calculate_reward(action_type)


        self.state["weight_history"].append(self.state["current_weight_kg"])
        self.state["stress_history"].append(self.state["stress_level"])
        self.state["energy_history"].append(self.state["energy_level"])

        self.state["time_of_day_slot"] += 1

        if self.state["time_of_day_slot"] >= self.slots_per_day:

            self.state["time_of_day_slot"] = 0
            self.state["day_of_episode"] += 1

            net_calories = self.state["daily_calories_intake"] - self.state["daily_calories_burned"]
            self.state["current_weight_kg"] += net_calories / 7700  

            self.state["daily_calories_intake"] = 0
            self.state["daily_calories_burned"] = 0
            self.state["daily_protein_intake"] = 0
            self.state["daily_fat_intake"] = 0
            self.state["daily_carbohydrate_intake"] = 0
            self.state["daily_saturated_fat_intake"] = 0
            self.state["daily_fiber_intake"] = 0

            done = self.state["day_of_episode"] >= self.days_per_episode


        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, done, info
