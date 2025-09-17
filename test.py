import os
import time
import numpy as np
import pybullet as p

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.biped_env import BipedEnv

# ----------- 创建环境 -----------
def make_env(render=False):
    def _init():
        env = BipedEnv(urdf_path="biped.urdf", render=render)
        return env
    return _init

# ----------- 加载模型和归一化器 -----------
model_path = "models/sac_biped"
vecnorm_path = "models/vecnorm.pkl"

venv = DummyVecEnv([make_env(render=True)])

# 加载归一化器
venv = VecNormalize.load(vecnorm_path, venv)
venv.training = False        # 验证时不要更新统计量
venv.norm_reward = False     # 验证时不要归一化奖励

# 加载训练好的模型
model = SAC.load(model_path, env=venv)

# ----------- 验证运行 -----------
obs = venv.reset()
for step in range(2000):  # 跑 2000 步
    action, _ = model.predict(obs, deterministic=True)  # 用确定性策略
    obs, reward, done, info = venv.step(action)
    venv.render()
    if done.any():
        print("Episode finished!")
        obs = venv.reset()
