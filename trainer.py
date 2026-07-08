import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import environment  # регистрация среды
from ppo_agent import PPOAgent
import os

BEST_REWARD_FILE = 'best_reward.txt'

def load_best_reward():
    if os.path.exists(BEST_REWARD_FILE):
        with open(BEST_REWARD_FILE, 'r') as f:
            return float(f.read().strip())
    return -float('inf')

def save_best_reward(reward):
    with open(BEST_REWARD_FILE, 'w') as f:
        f.write(str(reward))

# ---------- ФИНАЛЬНЫЕ ГИПЕРПАРАМЕТРЫ ----------
PARAMS = {
    'lr_actor': 0.0003538,
    'lr_critic': 0.0003047,
    'batch_size': 256,
    'gamma': 0.9985,
    'lambda_gae': 0.9717,
    'entropy_coef': 0.0204,
    'clip_epsilon': 0.1439,
    'max_policy_iters': 12,
    'max_value_iters': 6,
}

ENV_ID = 'sguschonka-racing-v0'
STATE_DIM = 7
ACTION_DIM = 2
EPISODES_PER_UPDATE = 20
STEPS_PER_EPISODE = 4096
TOTAL_UPDATES = 5000
CHECKPOINT_DIR = 'models/'
RENDER_TRAINING = False

# ---------- Построение моделей ----------
def build_actor_network(state_dim, action_dim):
    inputs = tf.keras.Input(shape=(state_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    mean = layers.Dense(action_dim, activation='tanh')(x)
    log_std = layers.Dense(action_dim)(x)
    return Model(inputs, [mean, log_std])

def build_critic_network(state_dim):
    inputs = tf.keras.Input(shape=(state_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    value = layers.Dense(1)(x)
    return Model(inputs, value)

# ---------- GAE ----------
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards)-1:
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        else:
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    return advantages, returns

# ---------- ОСНОВНОЙ ЦИКЛ ----------
def main():
    actor = build_actor_network(STATE_DIM, ACTION_DIM)
    critic = build_critic_network(STATE_DIM)

    agent = PPOAgent(
        actor_model=actor,
        critic_model=critic,
        epsilon=PARAMS['clip_epsilon'],
        target_kl_div=0.01,
        max_policy_iters=PARAMS['max_policy_iters'],
        max_value_iters=PARAMS['max_value_iters'],
        policy_lr=PARAMS['lr_actor'],
        value_lr=PARAMS['lr_critic'],
        batch_size=PARAMS['batch_size'],
        max_grad_norm=0.5,
        checkpoint_dir=CHECKPOINT_DIR
    )

    env = gym.make(ENV_ID, render_mode='human' if RENDER_TRAINING else None)

    best_reward = load_best_reward()
    print(f"Текущая лучшая награда: {best_reward:.2f}")

    for update in range(TOTAL_UPDATES):
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        episode_rewards = []

        for _ in range(EPISODES_PER_UPDATE):
            obs, _ = env.reset()
            done = False
            ep_rew = 0
            step = 0
            while not done and step < STEPS_PER_EPISODE:
                action, value, log_prob = agent.get_action_value(obs)
                action_np = action.numpy()[0]
                log_prob_np = log_prob.numpy()[0]

                next_obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated

                if RENDER_TRAINING:
                    env.render()

                states.append(obs)
                actions.append(action_np)
                rewards.append(reward)
                dones.append(done)
                values.append(value.numpy()[0][0])
                log_probs.append(log_prob_np)

                obs = next_obs
                ep_rew += reward
                step += 1
            episode_rewards.append(ep_rew)

        # Вычисляем последнее значение состояния
        last_obs = obs
        last_value = critic(tf.convert_to_tensor(last_obs[None, :], dtype=tf.float32)).numpy()[0][0]

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        old_log_probs = np.array(log_probs, dtype=np.float32)

        advantages, returns = compute_gae(rewards, values, dones, last_value,
                                          PARAMS['gamma'], PARAMS['lambda_gae'])

        agent.train_policy(states, actions, advantages, old_log_probs)
        agent.train_value(states, returns)

        avg_reward = np.mean(episode_rewards)
        print(f"Update {update:4d}, avg reward: {avg_reward:8.2f}, steps: {len(states)}")

        # ====== СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ======
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save_models()
            save_best_reward(best_reward)
            print(f"🔥 Новая лучшая модель! Награда: {best_reward:.2f}")

    print("Обучение завершено. Лучшая модель сохранена в", CHECKPOINT_DIR)
    env.close()

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    main()