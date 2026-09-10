"""Gymnasium environment for the 250 g, ten-servo reference biped.

Physics is isolated per client. The actor sees IMU-like signals, the clock and
command history (SG90 does not supply measured joint position). Simulator-only
velocities, contacts and joint states are used for rewards and diagnostics.
"""
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from envs.gait import gait_reference
from robot_config import ACTION_SCALE, JOINT_NAMES, ROBOT, ROOT


class BipedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": ROBOT.control_hz}
    JOINT_NAMES = JOINT_NAMES

    def __init__(self, urdf_path=None, render_mode=None, render=False, seed=None,
                 episode_len=600, target_speed=0.04, task="walk", domain_randomization=False):
        super().__init__()
        if render:
            render_mode = "human"
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if task not in ("stand", "walk"):
            raise ValueError("task must be stand or walk")
        if not isinstance(episode_len, int) or episode_len <= 0:
            raise ValueError("episode_len must be a positive integer")
        if not np.isfinite(target_speed) or not 0 <= target_speed <= 0.08:
            raise ValueError("target_speed must be between 0 and 0.08 m/s")
        self.urdf_path = Path(urdf_path).expanduser().resolve() if urdf_path else ROOT / "biped.urdf"
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        self.render_mode = render_mode
        self.episode_len = episode_len
        self.target_speed = float(target_speed) if task == "walk" else 0.0
        self.task = task
        self.domain_randomization = bool(domain_randomization)
        self.dt = 1 / ROBOT.control_hz
        self.sim_dt = 1 / ROBOT.physics_hz
        self.frame_skip = ROBOT.physics_hz // ROBOT.control_hz
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(JOINT_NAMES),), dtype=np.float32)
        # gravity(3), body gyro(3), phase sin/cos(2), speed(1), motor targets(10), last action(10)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(29,), dtype=np.float32)
        self._p = BulletClient(connection_mode=p.GUI if render_mode == "human" else p.DIRECT)
        self.cid = self._p._client
        self.closed = False
        self._needs_reset = True
        try:
            self.reset(seed=seed)
        except Exception:
            self.close()
            raise

    def _load_robot(self):
        self.robot_id = self._p.loadURDF(
            str(self.urdf_path), [0, 0, 0.25], useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
            | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        joint_info = [self._p.getJointInfo(self.robot_id, j)
                      for j in range(self._p.getNumJoints(self.robot_id))]
        mapping = {info[1].decode(): info for info in joint_info if info[2] == p.JOINT_REVOLUTE}
        if set(mapping) != set(JOINT_NAMES):
            raise ValueError(f"URDF requires these ten revolute joints: {JOINT_NAMES}")
        self.joint_indices = [mapping[name][0] for name in JOINT_NAMES]
        self.joint_name_to_index = dict(zip(JOINT_NAMES, self.joint_indices))
        self.low = np.asarray([mapping[n][8] for n in JOINT_NAMES])
        self.high = np.asarray([mapping[n][9] for n in JOINT_NAMES])
        if np.any(self.low >= self.high):
            raise ValueError("Every servo must have finite, ordered joint limits")
        self.force_limits = np.minimum(ROBOT.torque_limit, [mapping[n][10] for n in JOINT_NAMES])
        self.speed_limits = np.minimum(ROBOT.servo_speed, [mapping[n][11] for n in JOINT_NAMES])
        self.foot_indices = [self.joint_name_to_index[f"{side}_ankle_roll"] for side in ("left", "right")]
        self._p.setJointMotorControlArray(self.robot_id, self.joint_indices, p.VELOCITY_CONTROL,
                                         forces=[0.0] * len(JOINT_NAMES))
        self._p.changeDynamics(self.plane_id, -1, lateralFriction=0.8, restitution=0.0)
        friction = self.np_random.uniform(0.6, 1.0) if self.domain_randomization else 0.8
        mass_scale = self.np_random.uniform(0.9, 1.1) if self.domain_randomization else 1.0
        for index in [-1] + list(range(len(joint_info))):
            mass, _, inertia, *_ = self._p.getDynamicsInfo(self.robot_id, index)
            self._p.changeDynamics(self.robot_id, index, mass=mass * mass_scale,
                                   localInertiaDiagonal=(np.asarray(inertia) * mass_scale).tolist(),
                                   lateralFriction=friction, restitution=0.0,
                                   linearDamping=0.02, angularDamping=0.02)
        # The crossed-axis hip/ankle brackets overlap within one physical assembly.
        # Disable only these neighbours; left/right legs and torso/feet still collide.
        for side in ("left", "right"):
            hr, hp, knee, ap, ar = [self.joint_name_to_index[f"{side}_{n}"] for n in
                                    ("hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")]
            for a, b in ((-1, hp), (hp, ap), (knee, ar)):
                self._p.setCollisionFilterPair(self.robot_id, self.robot_id, a, b, 0)
        self.motor_strength = self.np_random.uniform(0.8, 1.0, 10) if self.domain_randomization else np.ones(10)
        self.motor_speed = self.speed_limits * (self.np_random.uniform(0.8, 1.0, 10)
                                              if self.domain_randomization else 1.0)
        self.latency_steps = int(self.np_random.integers(0, 2)) if self.domain_randomization else 0

    def _state(self):
        position, orientation = self._p.getBasePositionAndOrientation(self.robot_id)
        linear, angular = self._p.getBaseVelocity(self.robot_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        joints = self._p.getJointStates(self.robot_id, self.joint_indices)
        return {
            "position": np.asarray(position), "orientation": orientation,
            "gravity": rotation.T @ np.asarray([0.0, 0.0, -1.0]),
            "linear": np.asarray(linear), "gyro": rotation.T @ np.asarray(angular),
            "q": np.asarray([j[0] for j in joints]), "qd": np.asarray([j[1] for j in joints]),
        }

    def _get_obs(self, state=None):
        state = self._state() if state is None else state
        gyro, gravity = state["gyro"].copy(), state["gravity"].copy()
        if self.domain_randomization:
            gyro += self.np_random.normal(0, 0.015, 3)
            gravity += self.np_random.normal(0, 0.005, 3)
        phase = [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)]
        targets = 2 * (self.target_q - self.low) / (self.high - self.low) - 1
        return np.concatenate((gravity, gyro, phase, [self.target_speed / 0.08],
                               targets, self.last_action)).astype(np.float32)

    def _servo_step(self):
        """Solver-based position servo with finite torque and target velocity.

        Bullet gains are dimensionless solver parameters. The torque-speed
        envelope is an approximation; it requires bench calibration for SG90.
        """
        joints = self._p.getJointStates(self.robot_id, self.joint_indices)
        q = np.asarray([j[0] for j in joints])
        qd = np.asarray([j[1] for j in joints])
        direction = np.sign(self.target_q - q)
        limit = self.force_limits * self.motor_strength * np.clip(
            1 - direction * qd / self.motor_speed, 0, 1)
        for i, index in enumerate(self.joint_indices):
            self._p.setJointMotorControl2(
                self.robot_id, index, p.POSITION_CONTROL,
                targetPosition=float(self.target_q[i]),
                positionGain=ROBOT.servo_kp, velocityGain=ROBOT.servo_kd,
                force=float(limit[i]), maxVelocity=float(self.motor_speed[i]),
            )
        self._p.stepSimulation()
        self.last_torque = np.asarray([j[3] for j in self._p.getJointStates(
            self.robot_id, self.joint_indices)])

    def _contacts(self):
        points = self._p.getContactPoints(self.robot_id, self.plane_id)
        feet = np.asarray([any(c[3] == index and c[9] > 0.01 for c in points)
                           for index in self.foot_indices], dtype=float)
        body_contact = any(c[3] not in self.foot_indices and c[9] > 0.01 for c in points)
        return feet, body_contact

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.closed:
            raise RuntimeError("Environment is closed")
        self._p.resetSimulation()
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.81)
        self._p.setTimeStep(self.sim_dt)
        self._p.setPhysicsEngineParameter(numSolverIterations=80, deterministicOverlappingPairs=1)
        self.plane_id = self._p.loadURDF("plane.urdf")
        self._load_robot()
        self.phase = 0.0
        stand, _ = gait_reference(0.0, 0.0, walking=False)
        self.target_q = stand.copy()
        for index, angle in zip(self.joint_indices, stand):
            self._p.resetJointState(self.robot_id, index, float(angle), 0.0)
        # Use actual transformed foot AABBs, not a guessed torso height.
        minimum_z = min(self._p.getAABB(self.robot_id, f)[0][2] for f in self.foot_indices)
        z = 0.25 - minimum_z + 0.0005
        self._p.resetBasePositionAndOrientation(self.robot_id, [0, 0, z], [0, 0, 0, 1])
        self.last_torque = np.zeros(10)
        for _ in range(150):
            self._servo_step()
        state = self._state()
        feet, body_contact = self._contacts()
        if body_contact or state["gravity"][2] > -0.9 or feet.sum() < 2:
            raise RuntimeError("Reference robot cannot settle into two-foot support; check geometry/torque")
        self.standing_height = float(state["position"][2])
        self.start_position = state["position"].copy()
        self.last_action = np.zeros(10)
        self.pending_action = np.zeros(10)
        self.step_count = 0
        self.foot_transitions = np.zeros(2, dtype=int)
        self.previous_contacts = feet
        self.air_time = np.zeros(2)
        self.maximum_clearance = np.zeros(2)
        self._needs_reset = False
        if self.render_mode == "human":
            self._p.resetDebugVisualizerCamera(0.45, 40, -20, [0, 0, 0.10])
        return self._get_obs(state), {"mass_kg": ROBOT.total_mass}

    def step(self, action):
        if self.closed or self._needs_reset:
            raise RuntimeError("Call reset before stepping a closed/finished episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite array of shape (10,)")
        action = np.clip(action, -1, 1)
        applied = self.pending_action if self.latency_steps else action
        self.pending_action = action.copy()
        self.phase = (self.phase + self.dt / ROBOT.gait_period) % 1
        reference, expected_contacts = gait_reference(self.phase, self.target_speed, walking=self.task == "walk")
        # Ramp the gait prior on from the settled double-support pose.
        blend = min((self.step_count + 1) * self.dt / 0.5, 1.0)
        stand, _ = gait_reference(0, 0, walking=False)
        reference = stand + blend * (reference - stand)
        desired = np.clip(reference + applied * np.asarray(ACTION_SCALE), self.low, self.high)
        slew = self.motor_speed * self.dt
        self.target_q = np.clip(self.target_q + np.clip(desired - self.target_q, -slew, slew), self.low, self.high)
        energy = 0.0
        for _ in range(self.frame_skip):
            self._servo_step()
            energy += float(np.mean((self.last_torque / self.force_limits)**2)) / self.frame_skip
        self.step_count += 1
        state = self._state()
        feet, body_contact = self._contacts()
        for i, index in enumerate(self.foot_indices):
            clearance = max(0.0, self._p.getAABB(self.robot_id, index)[0][2])
            if feet[i] == 0:
                self.air_time[i] += self.dt
                self.maximum_clearance[i] = max(self.maximum_clearance[i], clearance)
            else:
                if self.air_time[i] >= 0.06 and self.maximum_clearance[i] >= 0.003:
                    self.foot_transitions[i] += 1
                self.air_time[i] = 0.0
                self.maximum_clearance[i] = 0.0
        self.previous_contacts = feet
        roll, pitch, yaw = p.getEulerFromQuaternion(state["orientation"])
        height = float(state["position"][2])
        terminated = bool(body_contact or height < 0.65 * self.standing_height
                          or abs(roll) > 0.7 or abs(pitch) > 0.7)
        truncated = bool(self.step_count >= self.episode_len and not terminated)
        # Fall detection uses raw state, never a clipped observation.
        vx, vy, vz = state["linear"]
        displacement = state["position"] - self.start_position
        speed_score = np.exp(-((vx - self.target_speed) / 0.035)**2)
        contact_score = float(np.mean(feet * expected_contacts + (1 - feet) * (1 - expected_contacts)))
        smooth_cost = float(np.mean((action - self.last_action)**2))
        height_cost = ((height - self.standing_height) / 0.03)**2
        components = {
            "velocity": 2.0 * float(speed_score),
            "progress": 1.0 * float(np.clip(vx / max(self.target_speed, 0.04), -1, 1)),
            "upright": 0.5 * float(-state["gravity"][2]),
            "contacts": 0.5 * contact_score,
            "pose": -0.15 * float(np.mean(((state["q"] - reference) / np.asarray(ACTION_SCALE))**2)),
            "drift": -0.3 * float((vy / 0.08)**2 + (displacement[1] / 0.10)**2 + yaw*yaw),
            "bounce": -0.1 * float((vz / 0.1)**2 + height_cost),
            "energy": -0.03 * energy,
            "smooth": -0.1 * smooth_cost,
        }
        reward = float(sum(components.values()))
        if terminated:
            reward = min(reward, 0.0) - 10.0
        self.last_action = action.copy()
        self._needs_reset = terminated or truncated
        info = {
            "vx": float(vx), "base_z": height, "x_distance": float(displacement[0]),
            "y_distance": float(displacement[1]), "elapsed_seconds": self.step_count * self.dt,
            "foot_contacts": feet.tolist(), "touchdowns": self.foot_transitions.tolist(),
            "is_fallen": terminated, "reward_terms": components,
        }
        if self.render_mode == "human":
            time.sleep(self.dt)
        return self._get_obs(state), reward, terminated, truncated, info

    def render(self):
        if self.closed:
            return None
        if self.render_mode == "rgb_array":
            position = self._p.getBasePositionAndOrientation(self.robot_id)[0]
            view = p.computeViewMatrixFromYawPitchRoll(position, 0.45, 45, -20, 0, 2)
            projection = p.computeProjectionMatrixFOV(55, 640 / 480, 0.01, 5)
            camera = self._p.getCameraImage(640, 480, view, projection, renderer=p.ER_TINY_RENDERER)
            return np.asarray(camera[2], dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
        return None

    def close(self):
        if not self.closed:
            self._p.disconnect()
            self.closed = True
