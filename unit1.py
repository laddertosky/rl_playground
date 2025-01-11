import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

ENVIRONMENT = 'LunarLander-v3'
MODEL_NAME = "ppo-LunarLander-v3"

def train_or_load():
    if os.path.exists(MODEL_NAME+".zip"):
        return PPO.load(MODEL_NAME)

    env = make_vec_env(ENVIRONMENT, n_envs=16)

    model = PPO(
        policy = 'MlpPolicy',
        env = env,
        n_steps = 1024,
        batch_size = 64,
        n_epochs = 4,
        gamma = 0.999,
        gae_lambda = 0.98,
        ent_coef = 0.01,
        verbose=1)

    model.learn(total_timesteps=1_000_000)

    # Save the model
    model.save(MODEL_NAME)
    return model

def eval(model):
    eval_env = Monitor(gym.make(ENVIRONMENT, render_mode='human'))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, render=True)

    # mean_reward=258.10 +/- 15.16840138874867
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

if __name__ == "__main__":
    model = train_or_load()
    eval(model)
