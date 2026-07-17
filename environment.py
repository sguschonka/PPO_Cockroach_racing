import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import os


class ImprovedLTYEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(ImprovedLTYEnv, self).__init__()
        self.render_mode = render_mode

        self.WIDTH = 1600
        self.HEIGHT = 860

        self.maps_folder = "src/maps/"

        # Параметры машины
        self.car_position_x = 948
        self.car_position_y = 1030
        self.max_steering_angle = 10
        self.max_speed = 25
        self.min_speed = 1
        self.max_acc = 1

        # Ограничение длины эпизода (например, 5000 шагов)
        self.max_steps_per_episode = 5000

        self.red = (225, 0, 0)
        self.green = (30, 152, 0)
        self.yellow = (255, 255, 0)
        self.grey = (64, 64, 64)

        low = np.array([-np.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        self.agent_state = None
        self.car_size_x = 32
        self.car_size_y = 32
        self.car_centre = [0, 0]
        self.car_radars = []
        self.dist = []
        self.agent_orientation_deg = 0.0

        if not pygame.get_init():
            pygame.init()

        if not os.path.exists("src/tarakan.png"):
            raise FileNotFoundError("Файл 'tarakan.png' не найден!")

        flags = 0
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("sguschonka-racing-v0")
        else:
            self.screen = pygame.display.set_mode(
                (self.WIDTH, self.HEIGHT), pygame.HIDDEN
            )

        self.tarakan_sprite = pygame.image.load("src/tarakan.png").convert_alpha()
        self.tarakan_sprite = pygame.transform.scale(
            self.tarakan_sprite, (self.car_size_x, self.car_size_y)
        )

        # ВАЖНО: карта загружается в reset, поэтому здесь просто создаём переменную
        self.track_map = None

        self.current_step = 0
        self.previous_laptime = 0

        # Загружаем первую случайную карту
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Загрузка случайной карты ---
        import os, random

        map_files = [f for f in os.listdir(self.maps_folder) if f.endswith(".png")]
        if not map_files:
            raise FileNotFoundError(f"Нет файлов .png в папке {self.maps_folder}")
        chosen_map = random.choice(map_files)
        self.track_map = pygame.image.load(
            os.path.join(self.maps_folder, chosen_map)
        ).convert()

        # --- Поиск стартовой позиции (по красному цвету) ---
        start_pos = None
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                color = self.track_map.get_at((x, y))
                if (color[0], color[1], color[2]) == self.red:
                    start_pos = (x - self.car_size_x / 2, y - self.car_size_y / 2)
                    break
            if start_pos is not None:
                break

        # --- Нормализация координат ---
        if start_pos is None:
            start_x, start_y = self.WIDTH / 2, self.HEIGHT / 2
        else:
            start_x, start_y = start_pos

        # Сохраняем реальные координаты для рендеринга и физики
        self.real_x = start_x
        self.real_y = start_y
        self.car_centre = [start_x + self.car_size_x / 2, start_y + self.car_size_y / 2]
        self.agent_orientation_deg = 0.0  # угол в градусах для лидаров

        # В агентское состояние записываем только: угол (0), скорость (0), лидары (0)
        self.agent_state = np.array(
            [
                0.0,  # orientation (рад)
                0.0,  # norm_vel
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # лидары
            ],
            dtype=np.float32,
        )

        self.current_step = 0
        self.car_radars = []
        self.dist = []

        return self.agent_state, {}

    def step(self, action):
        self.current_step += 1

        # Распаковываем состояние (без x,y)
        orientation, vel_norm, l1, l2, l3, l4, l5 = self.agent_state

        # Текущая реальная скорость
        real_vel = vel_norm * self.max_speed

        # Применяем действие (газ/тормоз)
        real_vel = real_vel + action[1] * self.max_acc
        real_vel = np.clip(real_vel, self.min_speed, self.max_speed)

        # Руль
        steer = np.clip(action[0], -1.0, 1.0)
        orientation = orientation + steer * np.pi * self.max_steering_angle / 180.0
        # нормализуем угол в [-pi, pi]
        orientation = (orientation + np.pi) % (2 * np.pi) - np.pi

        # Сохраняем предыдущие координаты для подсчёта пройденного расстояния
        prev_x, prev_y = self.real_x, self.real_y

        # Обновляем реальные координаты
        self.real_x = self.real_x + real_vel * np.cos(orientation)
        self.real_y = self.real_y + real_vel * np.sin(orientation)

        # Обновляем центр для рендеринга и лидаров
        self.car_centre = [
            self.real_x + self.car_size_x / 2,
            self.real_y + self.car_size_y / 2,
        ]
        self.agent_orientation_deg = orientation * 180 / np.pi

        # Вычисляем пройденное расстояние
        dist_moved = np.sqrt((self.real_x - prev_x) ** 2 + (self.real_y - prev_y) ** 2)

        # Обновляем лидары
        self.car_sensors()
        lidars = self.dist[:5]  # абсолютные расстояния (0-300)

        # ---- Награда ----
        reward = 0.0
        reward += dist_moved * 0.05
        if dist_moved < 0.05:
            reward -= 15.0

        # ---- Проверка выезда и финиша (как было) ----
        done = False
        truncated = False
        cx, cy = int(self.car_centre[0]), int(self.car_centre[1])
        if (1 < cx < self.WIDTH - 1) and (1 < cy < self.HEIGHT - 1):
            color = self.track_map.get_at((cx, cy))
            if (color[0], color[1], color[2]) == self.green:
                reward -= 100
                done = True
            if (color[0], color[1], color[2]) == self.grey:
                if self.current_step > self.previous_laptime:
                    reward += 200
                else:
                    reward -= 50
                self.previous_laptime = self.current_step
                done = True
        else:
            done = True

        if self.current_step >= self.max_steps_per_episode:
            truncated = True

        # ---- Новое наблюдение (только угол, скорость, лидары) ----
        norm_vel = real_vel / self.max_speed
        norm_lidars = [l / 300.0 for l in lidars]
        self.agent_state = np.array(
            [orientation, norm_vel] + norm_lidars, dtype=np.float32
        )

        terminated = done
        return self.agent_state, reward, terminated, truncated, {}

    def _get_normalized_state(self, x, y, orientation, velocity, lidars):
        norm_x = (x / self.WIDTH) * 2.0 - 1.0
        norm_y = (y / self.HEIGHT) * 2.0 - 1.0
        norm_vel = velocity / self.max_speed
        norm_lidars = [l / 300.0 for l in lidars]
        return np.array(
            [norm_x, norm_y, orientation, norm_vel] + norm_lidars, dtype=np.float32
        )

    def render(self):
        if self.render_mode == "human":
            pygame.event.pump()

            self.screen.blit(self.track_map, (0, 0))

            angle_deg = -self.agent_state[0] * 180 / np.pi
            rotated_sprite = pygame.transform.rotate(self.tarakan_sprite, angle_deg)

            car_center = (int(self.car_centre[0]), int(self.car_centre[1]))
            rect = rotated_sprite.get_rect(center=car_center)

            self.screen.blit(rotated_sprite, rect.topleft)

            for j in self.car_radars:
                pygame.draw.line(
                    self.screen, self.yellow, car_center, (int(j[0]), int(j[1])), 2
                )

            pygame.display.flip()

    def car_sensors(self):
        temp_points = []
        temp_dists = []
        for i in range(-90, 100, 45):
            point, dist = self.check_radar(i, self.track_map)
            temp_points.append(point)
            temp_dists.append(dist)
        self.car_radars = temp_points
        self.dist = temp_dists

    def check_radar(self, degree, game_map):
        length = 0
        angle_rad = math.radians(self.agent_orientation_deg + degree)
        cx, cy = self.car_centre

        while length < 300:
            x = int(cx + math.cos(angle_rad) * length)
            y = int(cy + math.sin(angle_rad) * length)

            if not (1 < x < self.WIDTH - 1) or not (1 < y < self.HEIGHT - 1):
                break

            color = game_map.get_at((x, y))
            if (color[0], color[1], color[2]) == self.green:
                break

            length += 1

        final_x = int(cx + math.cos(angle_rad) * length)
        final_y = int(cy + math.sin(angle_rad) * length)
        dist = int(math.sqrt((final_x - cx) ** 2 + (final_y - cy) ** 2))

        return (final_x, final_y), dist

    def close(self):
        pygame.quit()


gym.envs.register(
    id="sguschonka-racing-v0",
    entry_point="environment:ImprovedLTYEnv",
    disable_env_checker=True,
)
