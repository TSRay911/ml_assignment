import streamlit as st
import gymnasium as gym
from environment.environment3 import LifeStyleEnv
import numpy as np

# Streamlit page configuration
st.set_page_config(layout="wide")
st.title("Lifestyle Planner For Weight Management With Reinforcement Learning")


algorithm_option = st.selectbox("Select an algorithm", ("PPO", "DQN", "DreamerV3", "A2C"))
st.button("Run", type="primary", width="stretch")

with st.sidebar:
    st.header("User's Information:")

    initial_weight_= st.number_input("Enter Initial Weight (KG):",
                                     min_value=0.0,
                                     max_value=650.0, 
                                     step=5.0,
                                     value=70.0,
                                     )
    height = st.number_input("Enter Height (CM):",
                                     min_value=0.0,
                                     max_value=300.0, 
                                     step=5.0,
                                     value=170.0,
                                     )
    
    gender = st.radio(
        "What's your Gender",
        ["Male", "Female"],
    )

    target_bmi = st.number_input("Enter Target BMI:",
                                     min_value=10.0,
                                     max_value=80.0, 
                                     step=0.5,
                                     value=21.75,
                                     )
    
    work_mets = st.number_input("Work MET Level (1-12):",
                                     min_value=1.0,
                                     max_value=12.0, 
                                     step=0.5,
                                     value=2.0,
                                     )
    

current, timetable = st.tabs(["Current", "Plan"])

with current:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
with timetable:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    


