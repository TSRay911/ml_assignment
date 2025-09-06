from typing import Any, ClassVar, Optional, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.a2c.a2c import A2C  # subclass A2C
from stable_baselines3.common.policies import BasePolicy
from torch.nn import functional as F

from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer, MaskableDictRolloutBuffer
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

SelfMaskableA2C = TypeVar("SelfMaskableA2C", bound="MaskableA2C")


class MaskableA2C(A2C):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MaskableActorCriticPolicy,
        "CnnPolicy": MaskableActorCriticPolicy,
        "MultiInputPolicy": MaskableActorCriticPolicy,
    }

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        if not isinstance(self.policy, MaskableActorCriticPolicy):
            raise ValueError("Policy must subclass MaskableActorCriticPolicy")

        # choose rollout buffer class
        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = MaskableDictRolloutBuffer
            else:
                self.rollout_buffer_class = MaskableRolloutBuffer

        # instantiate rollout buffer
        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a MaskableRolloutBuffer.
        This mirrors MaskablePPO.collect_rollouts with A2C-style behavior.
        """
        assert isinstance(rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                if use_masking:
                    # returns an array shaped (n_envs, action_dim) or (n_envs,) depending on space
                    action_masks = get_action_masks(env)

                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_to_store = actions.reshape(-1, 1)
            else:
                actions_to_store = actions

            # Handle timeout bootstraping
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions_to_store,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        # compute last values for bootstrapping (no mask needed)
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()
        return True

    def predict(  # type: ignore[override]
        self,
        observation,
        state: Optional[tuple] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ):
        return self.policy.predict(observation, state, episode_start, deterministic, action_masks=action_masks)
