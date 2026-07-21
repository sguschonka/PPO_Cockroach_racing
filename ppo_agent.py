import tensorflow as tf
import numpy as np
import os


class PPOAgent:
    def __init__(
        self,
        actor_model,
        critic_model,
        epsilon,
        target_kl_div,
        max_policy_iters,
        max_value_iters,
        policy_lr,
        value_lr,
        batch_size=64,
        max_grad_norm=None,
        checkpoint_dir="models/",
        entropy_coef=0.01,
    ):

        self.checkpoint_dir = checkpoint_dir
        self.actor = actor_model
        self.critic = critic_model
        self.epsilon = epsilon
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_iters
        self.max_value_train_iters = max_value_iters
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef

        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=policy_lr, clipnorm=max_grad_norm
            )
        )
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=value_lr, clipnorm=max_grad_norm
            )
        )

    def save_models(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.actor.save(os.path.join(self.checkpoint_dir, "actor.keras"))
        self.critic.save(os.path.join(self.checkpoint_dir, "critic.keras"))

    def get_action_value(self, state):
        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        mean, log_std = self.actor(state_tensor)
        std = tf.exp(log_std)
        std = tf.clip_by_value(std, 1e-6, 1.0)

        value = self.critic(state_tensor)

        # Убираем clip_by_value для action, чтобы log_prob был честным
        action = tf.random.normal(shape=tf.shape(mean), mean=mean, stddev=std)
        log_prob = self.get_log_prob(mean, std, action)

        return action, value, log_prob

    def get_log_prob(self, mean, std, action):
        pre_sum = -0.5 * (
            ((action - mean) / std) ** 2 + 2 * tf.math.log(std) + tf.math.log(2 * np.pi)
        )
        return tf.reduce_sum(pre_sum, axis=-1)

    def train_policy(self, states, actions, advantages, old_log_probs):
        indices = np.arange(len(states))

        for _ in range(self.max_policy_train_iters):
            np.random.shuffle(indices)

            # Мини-батчинг
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                b_states = tf.convert_to_tensor(states[batch_idx], dtype=tf.float32)
                b_actions = tf.convert_to_tensor(actions[batch_idx], dtype=tf.float32)
                b_advantages = tf.convert_to_tensor(
                    advantages[batch_idx], dtype=tf.float32
                )
                b_old_log_probs = tf.convert_to_tensor(
                    old_log_probs[batch_idx], dtype=tf.float32
                )

                with tf.GradientTape() as tape:
                    mean, log_std = self.actor(b_states)
                    std = tf.exp(log_std)
                    std = tf.clip_by_value(std, 1e-6, 1.0)

                    new_log_probs = self.get_log_prob(mean, std, b_actions)
                    policy_ratio = tf.exp(new_log_probs - b_old_log_probs)

                    surrogate1 = policy_ratio * b_advantages
                    surrogate2 = (
                        tf.clip_by_value(
                            policy_ratio, 1 - self.epsilon, 1 + self.epsilon
                        )
                        * b_advantages
                    )

                    entropy = 0.5 * tf.reduce_sum(
                        tf.math.log(2 * np.pi * np.e) + 2 * tf.math.log(std), axis=-1
                    )

                    policy_loss = -tf.minimum(surrogate1, surrogate2) - self.entropy_coef * entropy
                    policy_loss = tf.reduce_mean(policy_loss)

                gradients = tape.gradient(policy_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(
                    zip(gradients, self.actor.trainable_variables)
                )

            # Early stopping по KL-divergence
            mean, log_std = self.actor(tf.convert_to_tensor(states, dtype=tf.float32))
            std = tf.exp(log_std)
            new_log_probs = self.get_log_prob(
                mean, std, tf.convert_to_tensor(actions, dtype=tf.float32)
            )
            kl_div = tf.reduce_mean(old_log_probs - new_log_probs)
            if kl_div > self.target_kl_div:
                break

        return policy_loss

    def train_value(self, states, returns):
        indices = np.arange(len(states))
        for _ in range(self.max_value_train_iters):
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                b_states = tf.convert_to_tensor(states[batch_idx], dtype=tf.float32)
                b_returns = tf.convert_to_tensor(returns[batch_idx], dtype=tf.float32)

                with tf.GradientTape() as tape:
                    values = self.critic(b_states)
                    value_loss = 0.5 * tf.reduce_mean(tf.square(values - b_returns))

                gradients = tape.gradient(value_loss, self.critic.trainable_variables)
                self.critic.optimizer.apply_gradients(
                    zip(gradients, self.critic.trainable_variables)
                )
        return value_loss