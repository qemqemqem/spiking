import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Box
import torch as th
import torch.nn as nn


class CustomBoxingObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.custom_vector_size = 5
        self.observation_space = gym.spaces.Dict({
            'original': self.observation_space,
            'custom': Box(low=-np.inf, high=np.inf, shape=(self.custom_vector_size,), dtype=np.float32)
        })

    def observation(self, observation):
        custom_vector = self.custom_logic()
        return {'original': observation, 'custom': custom_vector}

    def custom_logic(self):
        custom_vector = np.random.randn(self.custom_vector_size)
        return custom_vector


class CustomCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        self.original_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.custom_fc = nn.Sequential(
            nn.Linear(observation_space.spaces['custom'].shape[0], 64),
            nn.ReLU(),
        )

        self.final_fc = nn.Sequential(
            nn.Linear(self.features_dim + 64, self.features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        original_obs = observations['original'].permute(0, 3, 1, 2)
        original_features = self.original_cnn(original_obs)

        custom_obs = observations['custom']
        custom_features = self.custom_fc(custom_obs)

        combined_features = th.cat([original_features, custom_features], dim=1)
        return self.final_fc(combined_features)


def main():
    env = gym.make('Boxing-v0')
    env = CustomBoxingObservation(env)
    env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(
        features_extractor_class=CustomCnnExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = DQN('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=50000)
    model.save("dqn_boxing")

    del model  # Remove the trained model from memory
    model = DQN.load("dqn_boxing")

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
    env.close()


if __name__ == '__main__':
    main()
