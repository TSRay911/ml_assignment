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
import pandas as pd
from environment.MaskableA2C import MaskableA2C


# ---------- Environment Wrapper ----------
def make_env(is_eval: bool = False):
    env = LifeStyleEnv(st.session_state.initial_weight_kg, st.session_state.height_cm, 
                       st.session_state.gender, st.session_state.target_bmi, 
                       (0.0, 100.0), (0.0, 100.0), (0.0, 100.0), 365, st.session_state.work_mets)
    env = Monitor(env)
    if not is_eval:
        check_env(env, warn=True)
    return env

def run_episode_and_store(model_name):
    eval_env = st.session_state.eval_env
    model = st.session_state.current_model

    obs, info = eval_env.reset()
    unwrapped_env = eval_env.unwrapped

    done = False
    step_history = [] 
    reward_history = []

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

        reward_history.append(reward)

        step_history.append({
            "day": unwrapped_env.state["day_of_episode"] + 1,
            "timeslot": unwrapped_env.state["current_timeslot"],
            "action": map_action(int(action)),
            "event": event_applied,
            "reward": reward,
            "bmi": unwrapped_env.state["current_bmi"],
            "stress": unwrapped_env.state["current_stress_level"],
            "energy": unwrapped_env.state["current_energy_level"],
            "hunger": unwrapped_env.state["current_hunger_level"],
            "calories_in": unwrapped_env.state["daily_calories_intake"],
            "calories_out": unwrapped_env.state["daily_calories_burned"],
            "visual": visual
        })

    st.session_state.model_rewards[model] = reward_history

    df = pd.DataFrame(step_history)
    st.session_state.history_df[model_name] = df

    return df
 
def load_lottie_file(file_path: str):
    with open(file_path, "r") as file:
        return json.load(file)

def map_action(action):
    return action_map.get(action, "Unknown action")

action_map = {
    0: "Light meal",
    1: "Medium meal",
    2: "Heavy meal",
    3: "Light exercise",
    4: "Moderate exercise",
    5: "Intense exercise",
    6: "Relaxation",
    7: "Sleep/nap",
    8: "Event action"
}

# ---------- Initialize session_state ----------
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.algorithm_option = "PPO"
    st.session_state.initial_weight_kg = 70
    st.session_state.height_cm = 170
    st.session_state.gender = 0
    st.session_state.target_bmi = 21.75
    st.session_state.work_mets = 2.0 
    st.session_state.model_rewards = {}
    st.session_state.model_histories = {}
    st.session_state.history_df = {}
    st.session_state.current_model = MaskablePPO.load("environment/logs/ppo/ppo_best_model/best_model.zip")
    

if "eval_env" not in st.session_state:
    st.session_state.eval_env = make_env(is_eval=True)
# ---------- Page Config ----------
st.set_page_config(layout="wide")
st.title("Lifestyle Planner For Weight Management With Reinforcement Learning")



with st.sidebar:
    st.header("User's Information:")
    st.session_state.initial_weight_kg = st.number_input("Enter Initial Weight (KG):", min_value=0.0, max_value=650.0, step=5.0, value=70.0)
    st.session_state.height_cm = st.number_input("Enter Height (CM):", min_value=0.0, max_value=300.0, step=5.0, value=170.0)
    st.session_state.gender = st.radio("What's your Gender (0 - Male, 1 - Female)", ["0", "1"])
    st.session_state.target_bmi = st.number_input("Enter Target BMI:", min_value=10.0, max_value=80.0, step=0.5, value=21.75)
    st.session_state.work_mets = st.number_input("Work MET Level (1-12):", min_value=1.0, max_value=12.0, step=0.5, value=2.0)
    st.session_state.algorithm_option = st.selectbox("Select an algorithm", ("PPO", "DQN", "Dyna-Q", "A2C"))

    if st.button("Load Agent"):

        algorithms = ["PPO", "DQN", "Dyna-Q", "A2C"]
        
        for algorithm in algorithms:

            st.write(f"Running simulation for {algorithm}...")
            st.session_state.eval_env = make_env(is_eval=True)

            if st.session_state.algorithm_option == "PPO":
                st.session_state.current_model = MaskablePPO.load(
                    "environment/logs/ppo/ppo_best_model/best_model.zip"
                )
            elif st.session_state.algorithm_option == "A2C":
                st.session_state.current_model = MaskablePPO.load(
                    "environment/logs/a2c/a2c_best_model/best_model.zip"
                )
            elif st.session_state.algorithm_option == "A2C":
                st.session_state.current_model = MaskablePPO.load(
                    "environment/logs/ppo/ppo_best_model/best_model.zip"
                )
            else:
                st.session_state.current_model = MaskablePPO.load(
                    "environment/logs/ppo/ppo_best_model/best_model.zip"
                )

            st.session_state.model_histories[algorithm] = run_episode_and_store(algorithm)

        st.success("Environment initialized with user input ✅")

current, plan, performance_chart = st.tabs(["Simulation", "Plan", "Performance Comparision Chart"])


with current:
    st.subheader("Episode Simulation")

    if st.button("Replay Episode"):

        history = st.session_state.model_histories[st.session_state.algorithm_option].to_dict("records")

        col1, col2, col3, col4 = st.columns([1,5,3,3])

        with col2:
            visual_placeholder = st.empty()
        with col3:
            text_placeholder = st.empty()

        with col4:
            st.markdown("### User Information")
            st.write(f"Initial Weight: {st.session_state.initial_weight_kg} kg")
            st.write(f"Height: {st.session_state.height_cm} cm")
            st.write(f"Gender: {'Male' if int(st.session_state.gender) == 0 else 'Female'}")
            st.write(f"Target BMI: {st.session_state.target_bmi}")
            st.write(f"Work MET Level: {st.session_state.work_mets}")



        for step in history:
            with visual_placeholder.container():
                st.subheader(f"Current Algorithm: {st.session_state.algorithm_option}")
                st_lottie(load_lottie_file(step["visual"]),width=400,height=400)

            with text_placeholder.container():
                st.subheader(f"Day {step['day']}  | Timeslot {step['timeslot']}")
                st.write(f"Event: {step['event']} | Action: {step['action']}")
                st.write(f"Current BMI: {step['bmi']:.2f} | Stress: {step['stress']:.2f}")
                st.write(f"Energy: {step['energy']:.2f} | Hunger: {step['hunger']:.2f}")
                st.write(f"Calories Intake:{step['calories_in']:.2f} | Calories Burned {step['calories_out']:.2f}")
                st.write(f"Reward: {step['reward']:.2f}")


            time.sleep(1)

        st.header(f"Mean reward per step: {np.mean(st.session_state.model_rewards[st.session_state.algorithm_option])}")

with plan:
    with st.container():
        if "history_df" in st.session_state:
            st.subheader("Episode Data Table")
            st.write(f"Current algorithm: {st.session_state.algorithm_option}")
            if "history_df" in st.session_state and st.session_state.algorithm_option in st.session_state.history_df:
                st.dataframe(
                    st.session_state.history_df[st.session_state.algorithm_option],
                    width="stretch"
                )
            else:
                st.warning("No history available for the selected algorithm")

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
