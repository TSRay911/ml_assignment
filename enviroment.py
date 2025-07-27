import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class LifeStyleCoachEnv(gym.Env):
    
    def __init__(self, initial_weight: float = 80, height: float = 170, 
                 age: int = 21, gender: int = 0, target_weight: float = 65, 
                 stress_range: int = (1,10), days_per_episode: int = 28):
        
        super(LifeStyleCoachEnv, self).__init__()

        self.inital_weight = initial_weight
        self.height = height
        self.age = age
        self.gender = gender
        self.target_weight = target_weight
        self.stress_range = stress_range
        self.days_per_episode = days_per_episode

        self.action_space = spaces.Discrete(23)

