import time

import rospy
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from turtle_sim import TurtleEnv

n_env = 8


def make_env(index=1):
    def handle():
        env = TurtleEnv(index=index)
        return env

    return handle


if __name__ == "__main__":
    # PPO implementation from SB3
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

    env = SubprocVecEnv([make_env(index=i + 1) for i in range(n_env)])
    time.sleep(0.1)

    rospy.init_node("PPO")

    agent = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=1024,  # batch
        batch_size=128,  # mini-batch
        n_epochs=20,  #
        gamma=0.99,  # discount
        clip_range=0.2,  # advantage clip
        gae_lambda=0.95,
        verbose=1,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU, net_arch=dict(pi=[16, 16], vf=[32, 32])
        ),
    )
    # agent.learn(total_timesteps=25000)

    # agent.save("ppo_turtle")
    del agent # remove to demonstrate saving and loading
    agent = PPO.load("ppo_turtle")

    print("Start evaluation...")
    obs = env.reset()
    cumulative_reward = 0
    for _ in range(1000):
        action, _states = agent.predict(obs)
        obs, rewards, dones, info = env.step(action)
        cumulative_reward=+rewards
    print(f"Evaluation Return: {np.mean(cumulative_reward)}")

    env.close()