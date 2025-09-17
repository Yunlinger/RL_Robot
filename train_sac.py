import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.biped_env import BipedEnv
from stable_baselines3.common.monitor import Monitor
def make_env(render=False, seed: int = 0):
    def _init():
        env = BipedEnv(urdf_path="biped.urdf", render=render, seed=seed)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    seed = 0
    np.random.seed(seed)

    venv = DummyVecEnv([make_env(render=False, seed=seed)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    model = SAC(
        "MlpPolicy",
        venv,
        device="mps",  # 使用 auto，避免某些 mps 数值不稳定
        verbose=1,
        tensorboard_log="logs/",
        learning_starts=5000,
        batch_size=256,
        buffer_size=100_000,
        ent_coef="auto",  # 自动温度
        gamma=0.99,
        tau=0.005,
    )

    model.learn(total_timesteps=400_000)
    model.save("models/sac_biped")
    # 保存归一化统计量用于后续评估或继续训练
    venv.save("models/vecnorm.pkl")
