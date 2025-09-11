import os
import time
from typing import Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class BipedEnv(gym.Env):
    """
    二足机器人环境 (Gymnasium API):
    - 关节：left_hip, left_knee, right_hip, right_knee
    - 动作：目标角度增量控制
    - 状态：姿态 + 高度 + 关节状态 + 速度
    - 奖励：前向速度、直线保持、稳定性、动作平滑
    """

    metadata = {"render_modes": ["human"]}
    JOINT_NAMES = ["left_hip", "left_knee", "right_hip", "right_knee"]

    def __init__(
        self,
        urdf_path: str = "biped.urdf",
        render: bool = False,
        sim_timestep: float = 1.0 / 240.0,
        action_delta_limit: float = 0.10,
        kp: float = 0.9,
        kd: float = 0.05,
        max_force_hip: float = 60.0,
        max_force_knee: float = 40.0,
        episode_len: int = 2000,
        seed: int = 0,
        # reward shaping weights
        w_forward: float = 1.0,
        w_straight: float = 0.05,
        w_stability: float = 0.05,
        w_smooth: float = 0.01,
        survival_bonus: float = 0.02,
        fall_penalty: float = 2.0,
    ):
        super().__init__()
        self.urdf_path = urdf_path
        self.render_gui = render
        self.sim_timestep = sim_timestep
        self.action_delta_limit = float(action_delta_limit)
        self.kp = kp
        self.kd = kd
        self.max_force_hip = max_force_hip
        self.max_force_knee = max_force_knee
        self.episode_len = episode_len
        self.np_random = np.random.default_rng(seed)
        # reward weights
        self.w_forward = w_forward
        self.w_straight = w_straight
        self.w_stability = w_stability
        self.w_smooth = w_smooth
        self.survival_bonus = survival_bonus
        self.fall_penalty = fall_penalty

        # 连接仿真
        if render:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.sim_timestep)
        p.setGravity(0, 0, -9.8)

        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = None
        self.joint_name_to_index: Dict[str, int] = {}
        self.joint_limits: Dict[str, Tuple[float, float]] = {}
        self.target_q = None
        self.step_count = 0

        # 观测空间
        high = np.array(
            [np.pi, np.pi, np.pi, 2.0, 5.0, 5.0] + [np.pi] * 4 + [10.0] * 4,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # 动作空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.reset()

    # ------------------ 内部方法 ------------------
    def _load_robot(self):
        urdf_abspath = os.path.abspath(self.urdf_path)
        self.robot_id = p.loadURDF(urdf_abspath, [0, 0, 0.4], useFixedBase=False)

        self.joint_name_to_index.clear()
        self.joint_limits.clear()
        n_j = p.getNumJoints(self.robot_id)
        for j in range(n_j):
            info = p.getJointInfo(self.robot_id, j)
            name = info[1].decode("utf-8")
            jtype = info[2]
            lower, upper = info[8], info[9]
            if jtype == p.JOINT_REVOLUTE and name in self.JOINT_NAMES:
                self.joint_name_to_index[name] = j
                if lower >= upper:  # urdf 没写限位 → 给个默认值
                    lower, upper = (-1.2, 1.2) if "hip" in name else (0.0, 2.0)
                self.joint_limits[name] = (lower, upper)

        for idx in self.joint_name_to_index.values():
            p.setJointMotorControl2(self.robot_id, idx, controlMode=p.VELOCITY_CONTROL, force=0.0)

    def _place_camera(self):
        if self.render_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.3, cameraYaw=35, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.4]
            )

    def _set_pose(self):
        base_pos = [0, 0, 0.4]
        base_orn = p.getQuaternionFromEuler([0.0, 0.05, 0.0])
        p.resetBasePositionAndOrientation(self.robot_id, base_pos, base_orn)

        stand = {"left_hip": 0.0, "left_knee": 0.7, "right_hip": 0.0, "right_knee": 0.7}
        for name, q in stand.items():
            idx = self.joint_name_to_index[name]
            p.resetJointState(self.robot_id, idx, q, 0.0)

        self.target_q = np.array([stand[n] for n in self.JOINT_NAMES], dtype=np.float32)

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        lin_v, _ = p.getBaseVelocity(self.robot_id)
        base_z = pos[2]

        q_list, qd_list = [], []
        for name in self.JOINT_NAMES:
            idx = self.joint_name_to_index[name]
            js = p.getJointState(self.robot_id, idx)
            q_list.append(js[0])
            qd_list.append(js[1])

        obs = np.array(
            [roll, pitch, yaw, base_z, lin_v[0], lin_v[1]] + q_list + qd_list,
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _apply_action(self, action: np.ndarray):
        delta = np.clip(action, -1.0, 1.0) * self.action_delta_limit
        self.target_q = np.clip(self.target_q + delta, self._low_limits(), self._high_limits())
        for i, name in enumerate(self.JOINT_NAMES):
            idx = self.joint_name_to_index[name]
            maxF = self.max_force_knee if "knee" in name else self.max_force_hip
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(self.target_q[i]),
                positionGain=self.kp,
                velocityGain=self.kd,
                force=maxF,
            )

    def _compute_reward_done(self, obs, action):
        roll, pitch, yaw, base_z, vx, vy = obs[:6]
        r_forward = self.w_forward * np.tanh(vx)
        r_straight = -self.w_straight * (abs(yaw) + 0.2 * abs(vy))
        r_stability = -self.w_stability * (abs(roll) + abs(pitch))
        r_smooth = -self.w_smooth * np.linalg.norm(action, ord=1)
        r_alive = self.survival_bonus
        reward = float(r_alive + r_forward + r_straight + r_stability + r_smooth)

        terminated = False
        truncated = False
        if base_z < 0.18 or abs(roll) > 0.8 or abs(pitch) > 0.8:
            terminated = True
            reward -= float(self.fall_penalty)
        elif self.step_count >= self.episode_len:
            truncated = True
        info = {
            "vx": vx,
            "yaw": yaw,
            "base_z": base_z,
            # reward components for diagnostics
            "r_alive": r_alive,
            "r_forward": r_forward,
            "r_straight": r_straight,
            "r_stability": r_stability,
            "r_smooth": r_smooth,
        }
        return reward, terminated, truncated, info

    def _low_limits(self): return np.array([self.joint_limits[n][0] for n in self.JOINT_NAMES])
    def _high_limits(self): return np.array([self.joint_limits[n][1] for n in self.JOINT_NAMES])

    # ------------------ Gymnasium API ------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.sim_timestep)
        self.plane_id = p.loadURDF("plane.urdf")

        self._load_robot()
        self._place_camera()
        self._set_pose()
        self.step_count = 0

        noise = self.np_random.normal(0, 0.03, size=4).astype(np.float32)
        self.target_q = np.clip(self.target_q + noise, self._low_limits(), self._high_limits())

        obs = self._get_obs()
        return obs, {}


    def step(self, action):
        self.step_count += 1
        self._apply_action(action)

        for _ in range(4):
            p.stepSimulation()
            if self.render_gui:
                time.sleep(self.sim_timestep)

        obs = self._get_obs()
        reward, terminated, truncated, info = self._compute_reward_done(obs, action)
        return obs, reward, terminated, truncated, info


    def render(self): return None
    def close(self): p.disconnect()
