import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from training import (configure_sac_optimization, evaluate, load_bundle, make_vec_env,
                      prefill_replay_with_policy, reset_replay_buffer, save_bundle)


class TrainingTests(unittest.TestCase):
    def test_gradient_updates_bundle_roundtrip_and_resume(self):
        torch.set_num_threads(1)
        config = dict(episode_len=20, target_speed=0.04, task='walk',
                      domain_randomization=False, imu_model='bno085', seed=7)
        env = make_vec_env(config, training=True)
        loaded_env = None
        try:
            model = SAC('MlpPolicy', env, device='cpu', seed=7, learning_starts=16,
                        batch_size=16, buffer_size=256, policy_kwargs={'net_arch': [32, 32]})
            model.learn(48)
            self.assertGreater(model._n_updates, 0)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
            env.training = False
            env.seed(123)
            obs = env.reset()
            expected_action = model.predict(obs, deterministic=True)[0]
            with tempfile.TemporaryDirectory() as temp:
                bundle = Path(temp) / 'bundle'
                save_bundle(bundle, model, env, config, replay=True)
                selector = Path(temp) / 'best.json'
                selector.write_text('{"bundle": "bundle"}\n')
                loaded, loaded_env, _ = load_bundle(bundle, training=True)
                loaded_env.training = False
                loaded_env.seed(123)
                loaded_obs = loaded_env.reset()
                np.testing.assert_array_equal(obs, loaded_obs)
                np.testing.assert_array_equal(expected_action, loaded.predict(loaded_obs, deterministic=True)[0])
                count = loaded_env.obs_rms.count
                metrics = evaluate(loaded, loaded_env, episodes=1)
                self.assertEqual(count, loaded_env.obs_rms.count)
                self.assertEqual(len(metrics['episodes']), 1)
                self.assertEqual(loaded.replay_buffer.size(), model.replay_buffer.size())
                loaded_env.training = True
                updates = loaded._n_updates
                loaded.learn(16, reset_num_timesteps=False)
                self.assertEqual(loaded.num_timesteps, 64)
                self.assertGreater(loaded._n_updates, updates)
                selector_model, selector_env, _ = load_bundle(selector)
                try:
                    reset_replay_buffer(selector_model, selector_env, 256)
                    prefill_replay_with_policy(selector_model, selector_env, 64)
                    self.assertEqual(selector_model.replay_buffer.size(), 64)
                    configure_sac_optimization(
                        selector_model, learning_rate=1e-4, batch_size=32,
                        train_freq=2, gradient_steps=1, target_entropy=-5,
                        learning_starts=0,
                    )
                    self.assertEqual(selector_model.learning_starts, 0)
                    self.assertEqual(selector_model.batch_size, 32)
                    self.assertEqual(selector_model.train_freq.frequency, 2)
                finally:
                    selector_env.close()
        finally:
            env.close()
            if loaded_env is not None:
                loaded_env.close()


if __name__ == '__main__':
    unittest.main()
