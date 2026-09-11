import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from main import load_sbml_models, select_consortium_models, get_initial_params
from pathlib import Path

# Let's write a script to evaluate the latest model for one episode to see how long it survives.
