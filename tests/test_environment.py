import importlib
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env

from envs.biped_env import BipedEnv
from envs.gait import gait_reference
from robot_config import ROBOT, ROOT
from scripts.build_urdf import build


class EnvironmentTests(unittest.TestCase):
    def test_environment_contracts(self):
        with BipedEnv() as env:
            check_env(env)
            gym_check_env(env, skip_render_check=True)

    def test_seed_and_independent_physics_clients(self):
        with BipedEnv(domain_randomization=True) as first, BipedEnv() as second:
            obs1, _ = first.reset(seed=123)
            q1 = first._state()['q'].copy()
            obs2, _ = first.reset(seed=123)
            np.testing.assert_array_equal(obs1, obs2)
            np.testing.assert_array_equal(q1, first._state()['q'])
            second.reset(seed=0)
            state_before = second._state()['position'].copy()
            first.step(np.zeros(10))
            first.reset(seed=3)
            np.testing.assert_array_equal(state_before, second._state()['position'])
            first.close()
            self.assertTrue(np.isfinite(second.step(np.zeros(10))[0]).all())

    def test_invalid_actions_and_servo_limits(self):
        with BipedEnv(task='stand') as env:
            for invalid in (np.zeros(4), np.full(10, np.nan), np.full(10, np.inf)):
                with self.assertRaises(ValueError):
                    env.step(invalid)
            before = env.target_q.copy()
            env.step(np.full(10, 100))
            self.assertTrue(np.all(np.abs(env.target_q - before) <= env.motor_speed * env.dt + 1e-9))
            self.assertTrue(np.all(env.target_q >= env.low))
            self.assertTrue(np.all(env.target_q <= env.high))
            self.assertTrue(np.all(np.abs(env.last_torque) <= env.force_limits + 1e-9))

    def test_straight_walk_accepts_bounded_residuals(self):
        with BipedEnv(task='walk') as env:
            env.step(np.r_[np.ones(5), -np.ones(5)])
            np.testing.assert_array_equal(env.last_action, np.r_[np.ones(5), -np.ones(5)])

    def test_bno085_heading_feedback(self):
        with BipedEnv(task='stand', imu_model='bno085', seed=42) as env:
            position = env._state()['position']
            env._p.resetBasePositionAndOrientation(
                env.robot_id, position, env._p.getQuaternionFromEuler([0, 0, 0.5]))
            obs = env._get_obs()
            measured_heading = np.arctan2(obs[6], obs[7])
            self.assertAlmostEqual(measured_heading, 0.5 + env.imu_heading_bias, delta=0.02)
            self.assertEqual(obs.shape, (31,))

    def test_reference_supports_weight_for_full_episode(self):
        with BipedEnv(task='stand') as env:
            for _ in range(env.episode_len):
                obs, reward, terminated, truncated, info = env.step(np.zeros(10))
                self.assertFalse(terminated)
                self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward))
            self.assertTrue(truncated)
            self.assertLess(abs(info['x_distance']), 0.01)
            self.assertEqual(info['foot_contacts'], [1.0, 1.0])
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(10))

    def test_reference_walk_has_bounded_heading_and_lateral_drift(self):
        with BipedEnv(task='walk', imu_model='bno085') as env:
            for _ in range(env.episode_len):
                _, _, terminated, truncated, info = env.step(np.zeros(10))
                if terminated or truncated:
                    break
            self.assertFalse(terminated)
            self.assertLessEqual(info['max_abs_lateral_m'], 0.10)
            self.assertLessEqual(info['max_abs_heading_rad'], np.deg2rad(25))
            self.assertGreaterEqual(info['path_efficiency'], 0.80)
            self.assertTrue(np.isfinite(info['max_abs_pitch_deg']))
            self.assertTrue(np.isfinite(info['max_abs_roll_deg']))

    def test_reference_step_has_clearance_and_swing(self):
        phases = np.linspace(0.0, 1.0, 121, endpoint=False)
        knees = np.asarray([gait_reference(phase, 0.05)[0][2] for phase in phases])
        self.assertGreater(float(np.ptp(knees)), 0.35)
        self.assertGreaterEqual(ROBOT.foot_clearance, 0.010)

    def test_fall_separate_from_timeout(self):
        with BipedEnv(task='stand', episode_len=1) as env:
            _, _, terminated, truncated, _ = env.step(np.zeros(10))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            env.reset()
            env._p.resetBasePositionAndOrientation(env.robot_id, [0, 0, 0.02], [0, 0, 0, 1])
            _, reward, terminated, truncated, info = env.step(np.zeros(10))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info['is_fallen'])
            self.assertLessEqual(reward, -10)

    def test_randomized_resets(self):
        with BipedEnv(domain_randomization=True) as env:
            for seed in range(20):
                obs, _ = env.reset(seed=seed)
                self.assertTrue(np.isfinite(obs).all())

    def test_urdf_mass_geometry_and_render(self):
        self.assertEqual((ROOT / 'biped.urdf').read_text(), build())
        xml = ET.fromstring(build())
        mass = sum(float(x.attrib['value']) for x in xml.findall('./link/inertial/mass'))
        self.assertAlmostEqual(mass, 0.250)
        for link in xml.findall('link'):
            self.assertEqual(link.find('inertial/origin').attrib, link.find('collision/origin').attrib)
            inertia = link.find('inertial/inertia').attrib
            self.assertTrue(all(float(inertia[k]) > 0 for k in ('ixx', 'iyy', 'izz')))
        with BipedEnv(render_mode='rgb_array') as env:
            image = env.render()
            self.assertEqual(image.shape, (480, 640, 3))
            self.assertEqual(image.dtype, np.uint8)

    def test_replay_script_import_has_no_gui_side_effect(self):
        module = importlib.import_module('test')
        self.assertTrue(callable(module.main))


if __name__ == '__main__':
    unittest.main()
