import gymnasium as gym
import tensorflow as tf
import numpy as np
import os
import environment  # Для регистрации среды
import pygame

# ---------- Параметры ----------
ENV_ID = "sguschonka-racing-v0"
CHECKPOINT_DIR = "models/"
MAX_EPISODES = 10
MAX_STEPS = 5000

# ---------- Загрузка моделей ----------
print("Загрузка моделей...")
actor = tf.keras.models.load_model(os.path.join(CHECKPOINT_DIR, "actor.keras"))
print("Модели загружены.")

# ---------- Создаём среду с визуализацией ----------
env = gym.make(ENV_ID, render_mode="human")

# ---------- Тестирование ----------
for episode in range(MAX_EPISODES):
    obs, info = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done and step < MAX_STEPS:
        # Получаем действие (детерминированно, используем только mean)
        mean, _ = actor(tf.convert_to_tensor(obs[None, :], dtype=tf.float32))
        action = mean.numpy()[0]

        # Шаг в среде
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        obs = next_obs
        step += 1

        env.render()

    print(f"Эпизод {episode + 1}: награда = {total_reward:.2f}, шагов = {step}")

env.close()
print("Тестирование завершено.")
