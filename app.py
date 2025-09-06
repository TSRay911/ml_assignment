import streamlit as st
from training.environment3 import LifeStyleEnv
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from sb3_contrib.common.maskable.utils import get_action_masks
from streamlit_lottie import st_lottie
import time
import json
import pandas as pd
import altair as alt
from training.MaskableA2C import MaskableA2C
from stable_baselines3 import DQN
from training.dyna_q_lifestyle import DynaQLifestyle



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
            "Day": unwrapped_env.state["day_of_episode"] + 1,
            "Timeslot": unwrapped_env.state["current_timeslot"],
            "Action": map_action(int(action)),
            "Event": event_applied,
            "Reward": reward,
            "BMI": unwrapped_env.state["current_bmi"],
            "Stress": unwrapped_env.state["current_stress_level"],
            "Energy": unwrapped_env.state["current_energy_level"],
            "Hunger": unwrapped_env.state["current_hunger_level"],
            "Calories_intake": unwrapped_env.state["daily_calories_intake"],
            "Calories_burned": unwrapped_env.state["daily_calories_burned"],
            "Visual": visual
        })

    st.session_state.model_rewards[model] = reward_history

    df = pd.DataFrame(step_history)
    st.session_state.history_df[model_name] = df

    return df
 
def run_episode_and_store_dqn(model_name):
    eval_env = st.session_state.eval_env
    model = st.session_state.current_model

    obs, info = eval_env.reset()
    unwrapped_env = eval_env.unwrapped

    done = False
    step_history = [] 
    reward_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
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
            "Day": unwrapped_env.state["day_of_episode"] + 1,
            "Timeslot": unwrapped_env.state["current_timeslot"],
            "Action": map_action(int(action)),
            "Event": event_applied,
            "Reward": reward,
            "BMI": unwrapped_env.state["current_bmi"],
            "Stress": unwrapped_env.state["current_stress_level"],
            "Energy": unwrapped_env.state["current_energy_level"],
            "Hunger": unwrapped_env.state["current_hunger_level"],
            "Calories_intake": unwrapped_env.state["daily_calories_intake"],
            "Calories_burned": unwrapped_env.state["daily_calories_burned"],
            "Visual": visual
        })

    st.session_state.model_rewards[model] = reward_history

    df = pd.DataFrame(step_history)
    st.session_state.history_df[model_name] = df

    return df

def _visual(row):
                    if row["Event"] == "work":   return "assets/work.json"
                    if row["Event"] == "sleep":  return "assets/sleep.json"
                    a = str(row["Action"])
                    if a.startswith("meal"):      return "assets/eat.json"
                    if a.startswith("exercise"):  return "assets/exercise.json"
                    if a.startswith("rest"):      return "assets/rest.json"
                    if a == "skip":               return "assets/skip.json"
                    return "assets/skip.json"

def load_lottie_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
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
    st.session_state.initial_weight_kg = 85
    st.session_state.height_cm = 170
    st.session_state.gender = 0
    st.session_state.target_bmi = 21.75
    st.session_state.work_mets = 2.0 
    st.session_state.model_rewards = {}
    st.session_state.model_histories = {}
    st.session_state.history_df = {}
    st.session_state.current_model = MaskablePPO.load("training/logs/ppo/ppo_best_model_fined_tuned2/best_model.zip")
    

if "eval_env" not in st.session_state:
    st.session_state.eval_env = make_env(is_eval=True)

# ---------- Page Config ----------
st.set_page_config(layout="wide")
st.title("Lifestyle Planner For Weight Management With Reinforcement Learning")



with st.sidebar:
    st.header("User's Information:")
    st.session_state.initial_weight_kg = st.number_input("Enter Initial Weight (KG):", min_value=0.0, max_value=650.0, step=5.0, value=85.0)
    st.session_state.height_cm = st.number_input("Enter Height (CM):", min_value=0.0, max_value=300.0, step=5.0, value=170.0)
    st.session_state.gender = st.radio("What's your Gender (0 - Male, 1 - Female)", ["0", "1"])
    st.session_state.target_bmi = st.number_input("Target BMI:", min_value=10.0, max_value=80.0, step=0.5, value=21.7, disabled=True)
    st.session_state.work_mets = st.number_input("Work MET Level (1-12):", min_value=1.0, max_value=12.0, step=0.5, value=2.0)
    st.session_state.algorithm_option = st.selectbox("Select an algorithm", ("PPO", "DQN", "Dyna-Q", "A2C"))

    if st.button("Run Simulation"):

        algorithms = ["PPO", "DQN", "Dyna-Q", "A2C"]
        
        for algorithm in algorithms:

            st.write(f"Running simulation for {algorithm}...")
            st.session_state.eval_env = make_env(is_eval=True)

            if algorithm == "PPO":
                st.session_state.current_model = MaskablePPO.load(
                    "training/logs/ppo/ppo_best_model_fined_tuned2/best_model.zip"
                )
            elif algorithm == "A2C":
                st.session_state.current_model = MaskableA2C.load(
                    "training/logs/a2c/a2c_best_model/best_model.zip"
                )
            elif algorithm == "DQN":
                st.session_state.current_model = DQN.load(
                    "training/logs/dqn/dqn_best_model_fined_tuned5/best_model.zip"
                )
            else:
                st.session_state.current_model = DynaQLifestyle.load_q(
                    "training/saved_models/dyna_q_final_best.json"
                )

            if algorithm in ["PPO", "A2C"]:
                st.session_state.model_histories[algorithm] = run_episode_and_store(algorithm)
            elif algorithm == "Dyna-Q":

                rows, total_reward, a_hist = DynaQLifestyle.run_episode(
                    st.session_state.current_model, initial_weight_kg=st.session_state.initial_weight_kg, days_per_episode=365 , seed=999 
                )

                df = pd.DataFrame(
                    rows,
                    columns=[
                        "Day", "Timeslot", "Action", "Event",
                        "Reward", "BMI", "Stress", "Energy", 
                        "Hunger", "Calories_intake", "Calories_burned" 
                    ]
                )

                df["Visual"] = df.apply(_visual, axis=1)

                for col in ["BMI","Stress","Energy","Hunger","Calories_intake","Calories_burned","Reward"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
   
                st.session_state.history_df[algorithm] = df
                st.session_state.model_histories[algorithm] = df
                st.session_state.model_rewards[algorithm] = df["Reward"].tolist()

            else:
                st.session_state.model_histories[algorithm] = run_episode_and_store_dqn(algorithm)

        st.success("Environment initialized with user input ✅")

current, plan, performance_chart = st.tabs(["Simulation", "Plan", "Performance Comparision Chart"])


with current:
    st.subheader("Algorithm Simulation")

    if st.button("Replay Simulation"):

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
            st.write(f"Initial BMI: {st.session_state.initial_weight_kg/((st.session_state.height_cm / 100) ** 2):.2f}")
            st.write(f"Target BMI: {st.session_state.target_bmi}")
            st.write(f"Work MET Level: {st.session_state.work_mets}")



        for step in history:
            with visual_placeholder.container():
                st.subheader(f"Current Algorithm: {st.session_state.algorithm_option}")
                st_lottie(load_lottie_file(step["Visual"]),width=400,height=400)

            with text_placeholder.container():
                st.subheader(f"Day {step['Day']}  | Timeslot {step['Timeslot']}")
                st.write(f"Event: {step['Event']} | Action: {step['Action']}")
                st.write(f"Current BMI: {step['BMI']:.2f} | Stress: {step['Stress']:.2f}")
                st.write(f"Energy: {step['Energy']:.2f} | Hunger: {step['Hunger']:.2f}")
                st.write(f"Calories Intake:{step['Calories_intake']:.2f} | Calories Burned {step['Calories_burned']:.2f}")
                st.write(f"Reward: {step['Reward']:.2f}")


            time.sleep(0.5)

with plan:
    with st.container():
        if "history_df" in st.session_state:
            st.subheader("Episode Data Table")
            st.write(f"Current algorithm: {st.session_state.algorithm_option}")
            if st.button("Show Latest Data"):
                if ("history_df" in st.session_state and st.session_state.algorithm_option in st.session_state.history_df):
                    st.dataframe(st.session_state.history_df[st.session_state.algorithm_option],width="stretch")
                else:
                    st.warning("No history available for the selected algorithm")

# Main content
with performance_chart:
    st.header("Algorithm Performance Comparison")
    if not st.session_state.model_histories:
        st.info("Run simulations from the sidebar to see performance charts.")
    else:
        # Get a list of all available algorithms
        available_algorithms = list(st.session_state.model_histories.keys())
        
        # Checkbox for selecting which algorithms to display
        selected_algorithms = st.multiselect(
            "Select algorithms to display on charts:",
            options=available_algorithms,
            default=available_algorithms
        )
        
        if not selected_algorithms:
            st.warning("Please select at least one algorithm to display.")
        else:
            all_histories = []
            for algorithm in selected_algorithms:
                df = st.session_state.model_histories[algorithm]
                temp_df = df.copy()
                temp_df['algorithm'] = algorithm
                temp_df['step'] = range(len(temp_df))
                all_histories.append(temp_df)
            
            combined_df = pd.concat(all_histories, ignore_index=True)

            episode_length_df = (
                combined_df.groupby('algorithm', as_index=False)['Day']
                .max()
                .rename(columns={'Day': 'episode_length'})
            )

            # Cumulative Reward Chart
            st.subheader("Cumulative Reward Comparison")
            combined_df['cumulative_reward'] = combined_df.groupby('algorithm')['Reward'].cumsum()

            reward_chart = (alt.Chart(combined_df)
                            .mark_line()
                            .encode(
                                x=alt.X('step', title='Step'),
                                y=alt.Y('cumulative_reward', title='Cumulative Reward'),
                                color=alt.Color('algorithm', title='Algorithm'),
                                tooltip=['step', 'cumulative_reward', 'algorithm']
                                ).properties(
                                    title='Cumulative Reward Per Step Across Algorithms')
                        )
            st.altair_chart(reward_chart, use_container_width=True)

            # Mean Reward Bar Chart
            mean_rewards_df = (
                combined_df.groupby('algorithm', as_index=False)['Reward']
                .mean()
                .rename(columns={'Reward': 'mean_reward'})
            )
            
            st.subheader("Mean Reward")
            mean_reward_bar = (alt.Chart(mean_rewards_df)
                            .mark_bar()
                            .encode(
                                x=alt.X('algorithm', title='Algorithm'),
                                y=alt.Y('mean_reward', title='Mean Reward'),
                                color=alt.Color('algorithm', legend=None),
                                tooltip=['algorithm', 'mean_reward']
                            )
                            .properties(
                                title='Average Reward per Step (Mean)'
                            ))
            st.altair_chart(mean_reward_bar, use_container_width=True)

            st.subheader("Episode Length")
            ael_chart = (
                alt.Chart(episode_length_df)
                .mark_bar()
                .encode(
                    x=alt.X('algorithm', title='Algorithm'),
                    y=alt.Y('episode_length', title='Day Reached'),
                    color=alt.Color('algorithm', legend=None),
                    tooltip=['algorithm', 'episode_length']
                )
                .properties(
                    title='Episode Length'
                )
            )
            st.altair_chart(ael_chart, use_container_width=True)


            # BMI Trajectory Chart
            st.subheader("BMI Trajectory Comparison")
            bmi_chart = (alt.Chart(combined_df)
                         .mark_line()
                         .encode(
                             x=alt.X('step', title='Step'),
                             y=alt.Y('BMI', title='BMI', scale=alt.Scale(zero=False)),
                             color=alt.Color('algorithm', title='Algorithm'),
                             tooltip=['step', 'BMI', 'algorithm']
                             ).properties(
                                 title='BMI Trajectory Across Algorithms')
                        )
            st.altair_chart(bmi_chart, use_container_width=True)

            # Stress Level Chart
            st.subheader("Stress Level Comparison")
            stress_chart = (alt.Chart(combined_df)
                            .mark_line(strokeWidth=3.0)
                            .encode(
                                x=alt.X('step', title='Step'),
                                y=alt.Y('Stress', title='Stress Level', scale=alt.Scale(zero=False)),
                                color=alt.Color('algorithm', title='Algorithm'),
                                tooltip=['step', 'Stress', 'algorithm']
                            ).properties(
                                title='Stress Level Across Algorithms').interactive()
                        )
            st.altair_chart(stress_chart, use_container_width=True)
            
            # Calories Chart
            st.subheader("Calories Intake (Kcal)")
            calories_intake_chart = (alt.Chart(combined_df)
                                .mark_line()
                                .encode(
                                    x=alt.X('step', title='Steps'),
                                    y=alt.Y('Calories_intake', title='Calories Intake (Kcal)'),
                                    color=alt.Color('algorithm', title='Algorithm'),
                                    tooltip=['step', 'Calories_intake', 'algorithm']
                                ).properties(
                                    title='Daily Calories Intake'
                                ).interactive()
                            )
            st.altair_chart(calories_intake_chart, use_container_width=True)

            # Calories Chart
            st.subheader("Calories Burned (Kcal)")
            calories_expenditure_chart = (alt.Chart(combined_df)
                                .mark_line()
                                .encode(
                                    x=alt.X('step', title='Steps'),
                                    y=alt.Y('Calories_burned', title='Calories Burned (Kcal)'),
                                    color=alt.Color('algorithm', title='Algorithm'),
                                    tooltip=['step', 'Calories_burned', 'algorithm']
                                ).properties(
                                    title='Daily Calories Expenditure'
                                ).interactive()
                            )
            st.altair_chart(calories_expenditure_chart, use_container_width=True)

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
