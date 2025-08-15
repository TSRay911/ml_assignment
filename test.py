import gymnasium as gym
from enviroment import LifeStyleCoachEnv
from gymnasium.wrappers import FlattenObservation



env = LifeStyleCoachEnv()
env = FlattenObservation(env)
obs, info = env.reset(seed=42)


print("Action space:", env.action_space)
print("Observation space:", env.observation_space)


for i in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward, terminated, truncated)
    if terminated or truncated:
        obs, info = env.reset()

