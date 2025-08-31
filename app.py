import streamlit as st
from environment.environment3 import LifeStyleEnv
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from sb3_contrib.common.maskable.utils import get_action_masks
from streamlit_lottie import st_lottie
import time
import json

# ---------- Environment Wrapper ----------
def make_env(is_eval: bool = False):
    env = LifeStyleEnv()
    env = Monitor(env)
    if not is_eval:
        check_env(env, warn=True)
    return env

def run_episode_and_store():
    eval_env = st.session_state.eval_env
    model = st.session_state.current_model

    obs, info = eval_env.reset()
    unwrapped_env = eval_env.unwrapped

    done = False
    step_history = [] 

    while not done:
        action_masks = get_action_masks(unwrapped_env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

        timeslot_applied = unwrapped_env.state["current_timeslot"] - 1
        timeslot_applied = max(timeslot_applied, 0)
        event_applied = unwrapped_env.daily_schedule[timeslot_applied]

        if event_applied == "work":
            visual = "assets/work.json"
        elif event_applied == "sleep":
            visual = "assets/sleep.json"
        else:
            if action in [0, 1, 2]:
                visual = "assets/eat.json"
            elif action in [3, 4, 5]:
                visual = "assets/exercise.json"
            elif action in [6, 7]:
                visual = "assets/rest.json"
            else:
                visual = "assets/skip.json"

        step_history.append({
            "day": unwrapped_env.state['day_of_episode'],
            "timeslot": unwrapped_env.state['current_timeslot'],
            "action": action,
            "event": event_applied,
            "reward": reward,
            "bmi": unwrapped_env.state['current_bmi'],
            "stress": unwrapped_env.state['current_stress_level'],
            "energy": unwrapped_env.state['current_energy_level'],
            "hunger": unwrapped_env.state['current_hunger_level'],
            "calories_in": unwrapped_env.state['daily_calories_intake'],
            "calories_out": unwrapped_env.state['daily_calories_burned'],
            "visual": visual
        })

    st.session_state.history = step_history
    
def load_lottie_file(file_path: str):
    with open(file_path, "r") as file:
        return json.load(file)

action_map = {
    0: "Light meal",
    1: "Medium meal",
    2: "Heavy meal",
    3: "Light exercise",
    4: "Moderate exercise",
    5: "Intense exercise",
    6: "Relaxation",
    7: "Sleep/nap",
    8: "Skip / do nothing"
}

# ---------- Initialize session_state ----------
if "eval_env" not in st.session_state:
    st.session_state.eval_env = make_env(is_eval=True)

if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.algorithm_option = "Placeholder"
    st.session_state.episode_rewards = []
    st.session_state.initial_weight_kg = 70
    st.session_state.height_cm = 170
    st.session_state.gender = 0
    st.session_state.target_bmi = 21.75
    st.session_state.work_mets = 2.0
    st.session_state.current_model = None   
    st.session_state.history = None
    
# ---------- Page Config ----------
st.set_page_config(layout="wide")
#st.title("Lifestyle Planner For Weight Management With Reinforcement Learning")

if st.session_state.algorithm_option == "PPO":
    if st.session_state.current_model is None: 
        st.session_state.current_model = MaskablePPO.load(
            "agent/ppo_lifestylecoach_best_entropy.zip"
        )

with st.sidebar:
    st.header("User's Information:")
    st.session_state.initial_weight_kg = st.number_input("Enter Initial Weight (KG):", min_value=0.0, max_value=650.0, step=5.0, value=70.0)
    st.session_state.height_cm = st.number_input("Enter Height (CM):", min_value=0.0, max_value=300.0, step=5.0, value=170.0)
    st.session_state.gender = st.radio("What's your Gender (0 - Male, 1 - Female)", ["0", "1"])
    st.session_state.target_bmi = st.number_input("Enter Target BMI:", min_value=10.0, max_value=80.0, step=0.5, value=21.75)
    st.session_state.work_mets = st.number_input("Work MET Level (1-12):", min_value=1.0, max_value=12.0, step=0.5, value=2.0)
    st.session_state.algorithm_option = st.selectbox("Select an algorithm", ("PPO", "DQN", "Dyna-Q", "A2C"))

    if st.button("Load Agent"):
        run_episode_and_store()


current, performance_chart = st.tabs(["Simulation", "Performance Comparision Chart"])

with current:
    st.subheader("Episode Simulation")

    if "history" in st.session_state and st.button("Replay Episode"):
        col1, col2 = st.columns([3, 10])

        with col1:
            visual_placeholder = st.empty()
        with col2:
            text_placeholder = st.empty()

        for step in st.session_state.history:
            with visual_placeholder.container():
                st_lottie(load_lottie_file(step["visual"]))

            with text_placeholder.container():
                st.subheader(f"Day {step['day'] + 1} | Timeslot {step['timeslot']}")
                st.write(f"Action: {step['action']} | Event: {step['event']} | Reward: {step['reward']:.2f}")
                st.write(
                    f"BMI: {step['bmi']:.2f}, Stress: {step['stress']:.2f}, "
                    f"Energy: {step['energy']:.2f}, Hunger: {step['hunger']:.2f}"
                )

            time.sleep(1)  

with performance_chart:
    with st.container():
        st.write("This is inside the container")

# CSS code
st.markdown(
    """
    <style>
        .stElementContainer{
            width:100%;
        }

        .stButton > button{
            font-size: 20px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)
