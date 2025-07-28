import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
from actions import Action

class LifeStyleCoachEnv(gym.Env):
    
    def __init__(self, initial_weight_kg: float = 80.0, height_cm: float = 170.0, 
                 age: int = 21, gender: int = 0, target_bmi: float = 20.0, 
                 stress_range: tuple[float, float] = (1.0, 10.0), days_per_episode: int = 28):
        
        super(LifeStyleCoachEnv, self).__init__()

        # Internel state variables
        self.initial_weight_kg = initial_weight_kg
        self.height_cm = height_cm
        self.age = age
        self.gender = gender
        self.target_bmi = target_bmi
        self.days_per_episode = days_per_episode
        self.min_stress_level, self.max_stress_level = stress_range

        # Define the range of the values in the observation space
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
                "day_of_episode": gym.spaces.Box(low=0, high=self.days_per_episode, shape=(), dtype=np.int32)                                
            }
        )

        # Dynamic state variables
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
            "stress_history": []
        }

        self.action_space = spaces.Discrete(23)

    def _calculate_bmi(self):
        if self.height_cm <= 0:
            return 0.0

        return self.state["current_weight_kg"] / ((self.height_cm / 100) ** 2)
    
    def _get_obs(self):

        current_bmi = self._calculate_bmi()

        return{
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
            "day_of_episode": np.array(self.state["day_of_episode"], dtype=np.int32) 
        }
    
    def _get_info(self):
        current_bmi = self._calculate_bmi() 
        bmi_deviation = abs(current_bmi - self.target_bmi)

        return {
            "current_bmi": current_bmi,
            "target_bmi": self.target_bmi,
            "bmi_deviation": bmi_deviation,
            "net_calories_today": self.state['daily_calories_intake'] - self.state['daily_calories_burned'],
        }

        



