import random

import gymnasium as gym
import imageio
import numpy as np
import pickle5 as pickle
from gymnasium.spaces import Discrete
from tqdm import tqdm


class Hyperparameters:
    n_training_episodes = 100000  # Total training episodes
    # Training parameters
    learning_rate = 0.005  # Learning rate

    # Evaluation parameters
    n_eval_episodes = 100  # Total number of test episodes

    # Environment parameters
    max_steps = 99  # Max steps per episode
    gamma = 0.99  # Discounting rate

    eval_seed = []

    # Exploration parameters
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.05  # Minimum exploration probability
    decay_rate = 0.00005  # Exponential decay rate for exploration prob

class LakeHP(Hyperparameters):
    n_training_episodes = 100000
    learning_rate = 0.005  
    decay_rate = 0.00005  

class TaxiHP(Hyperparameters):
    n_training_episodes = 50000
    learning_rate = 0.01
    decay_rate = 0.0005
    eval_seed = [
        16,
        54,
        165,
        177,
        191,
        191,
        120,
        80,
        149,
        178,
        48,
        38,
        6,
        125,
        174,
        73,
        50,
        172,
        100,
        148,
        146,
        6,
        25,
        40,
        68,
        148,
        49,
        167,
        9,
        97,
        164,
        176,
        61,
        7,
        54,
        55,
        161,
        131,
        184,
        51,
        170,
        12,
        120,
        113,
        95,
        126,
        51,
        98,
        36,
        135,
        54,
        82,
        45,
        95,
        89,
        59,
        95,
        124,
        9,
        113,
        58,
        85,
        51,
        134,
        121,
        169,
        105,
        21,
        30,
        11,
        50,
        65,
        12,
        43,
        82,
        145,
        152,
        97,
        106,
        55,
        31,
        85,
        38,
        112,
        102,
        168,
        123,
        97,
        21,
        83,
        158,
        26,
        80,
        63,
        5,
        81,
        32,
        11,
        28,
        148,
    ]  # The evaluation seed of the environment

def greedy_policy(q_table, state):
    return np.argmax(q_table[state][:])

def train_policy(action_space, q_table, state, epsilon):
    rand = random.uniform(0, 1)

    if rand < epsilon:
        return action_space.sample()
    else:
        return greedy_policy(q_table, state)

def train(env: gym.Env, params: Hyperparameters = Hyperparameters()):
    if not isinstance(env.observation_space, Discrete):
        raise RuntimeError("Incompatible environment usage (state should be discrete).")
    if not isinstance(env.action_space, Discrete):
        raise RuntimeError("Incompatible environment usage (action should be discrete).")

    state_size = env.observation_space.n
    action_size = env.action_space.n

    q_table = np.zeros((state_size, action_size))
    for episode in tqdm(range(params.n_training_episodes), desc="Training"):
        epsilon = params.min_epsilon + (params.max_epsilon - params.min_epsilon) * np.exp(-params.decay_rate * episode)

        observation, _ = env.reset()

        for _ in range(params.max_steps):
            action = train_policy(env.action_space, q_table, observation, epsilon)
            new_observation, reward, terminated, truncated, _ = env.step(action)

            diff = reward + params.gamma * np.max(q_table[new_observation][:]) - q_table[observation][action]
            q_table[observation][action] = q_table[observation][action] + params.learning_rate * diff

            if terminated or truncated:
                break

            observation = new_observation
    return q_table

def eval(env: gym.Env, q_table, params: Hyperparameters = Hyperparameters()):
    rewards = []
    for episode in tqdm(range(params.n_eval_episodes), desc="Evaluating"):
        observation, _ = env.reset(seed=params.eval_seed[episode % len(params.eval_seed)]) if params.eval_seed else env.reset()

        episode_reward = 0
        for _ in range(params.max_steps):
            action = greedy_policy(q_table, observation)

            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)

            if terminated or truncated:
                break

        rewards.append(episode_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return mean_reward, std_reward

def fronzen_lake():
    map = "4x4"
    env_id = "FrozenLake-v1"
    env = gym.make(env_id, render_mode=None, map_name=map, is_slippery=True)

    params = LakeHP()
    q_table = train(env, params)

    # env = gym.make(env_id, render_mode="human", map_name=map, is_slippery=True)
    # env.metadata["render_fps"] = 60
    mean_reward, std_reward = eval(env, q_table, params)
    print(f"Env: {env_id}: {mean_reward = } +- {std_reward = }, (lower = {mean_reward - std_reward})")
    # env.render()

    '''
        Env: FrozenLake-v1: mean_reward = 0.81 +- std_reward = 0.3923009049186606, (lower = 0.41769909508133946)
    '''

def taxi():
    env_id = "Taxi-v3"
    env = gym.make(env_id, render_mode=None)

    params = TaxiHP()
    q_table = train(env, params)

    # env = gym.make(env_id, render_mode="human")
    # env.metadata["render_fps"] = 60
    mean_reward, std_reward = eval(env, q_table, params)
    print(f"Env: {env_id}: {mean_reward = } +- {std_reward = }, (lower = {mean_reward - std_reward})")
    # env.render()

    '''
        Env: Taxi-v3: mean_reward = 7.56 +- std_reward = 2.706732347314747, (lower = 4.853267652685252)
    '''

if __name__ == "__main__":
    fronzen_lake()
    taxi()
