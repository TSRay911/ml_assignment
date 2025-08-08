import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class LifeStyleCoachEnv(gym.Env):

    # Static parameters to initialize environment
    def __init__(self, inital_weight_kg: float = 80, height_cm: float = 165, age: int = 22,
                 gender: int = 0, target_bmi: float = 20, stress_range: tuple[float, float] = (1.0, 10.0),
                 energy_range: tuple[float, float] = (0.0, 10.0), days_per_episode: int = 28):
    
        super(LifeStyleCoachEnv, self).__init__()

        # Static variables
        self.initial_weight_kg = inital_weight_kg
        self.height_cm = height_cm
        self.age = age
        self.gender = gender
        self.target_bmi = target_bmi
        self.stress_range = stress_range
        self.energy_range = energy_range
        self.days_per_episode = days_per_episode

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
            "stress_history": []
        }

        self.slots_per_day = 24
        self.daily_schedule = ['agent_controlled_action'] * self.time_slots_per_day

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
                "day_of_episode": gym.spaces.Box(low=0, high=self.days_per_episode, shape=(), dtype=np.int32)
            }
        )

        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(3), 
            "action_details": spaces.MultiDiscrete([
                4, 4, 4, 4, 4, 4, # Meal choices
                4,                # Exercise choices
                2                 # Rest choices
            ])
        })

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
            "day_of_episode": np.array(self.state["day_of_episode"], dtype=np.int32) 
        }
    
    def _calculate_bmi(self):
        
        if self.height_cm <= 0:
            return 0.0

        return self.state["current_weight_kg"] / ((self.height_cm / 100) ** 2)
    
   


