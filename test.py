from environment.environment3 import LifeStyleEnv
from gymnasium.wrappers import FlattenObservation

env = LifeStyleEnv()
env = FlattenObservation(env)
obs, info = env.reset(seed=42)

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)


for i in range(1):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward, terminated, truncated)
    if terminated or truncated:
        obs, info = env.reset()

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)