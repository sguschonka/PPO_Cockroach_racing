import gymnasium as gym
import environment

env = gym.make("sguschonka-racing-v0", render_mode="human")

for episode in range(1):  # Достаточно одного эпизода
    obs, info = env.reset()
    print(f"\n--- Эпизод {episode + 1} ---")

    # Точка спавна
    x, y = int(env.unwrapped.car_centre[0]), int(env.unwrapped.car_centre[1])
    color = env.unwrapped.track_map.get_at((x, y))
    print(f"✅ СПАВН: Центр={env.unwrapped.car_centre}, Цвет={color}")

    # Делаем 1 шаг с нулевым газом (машина просто едет по инерции/минимальной скорости)
    action = [0.0, 0.0]
    obs, reward, terminated, truncated, info = env.step(action)

    # Точка ПОСЛЕ шага
    x_new, y_new = int(env.unwrapped.car_centre[0]), int(env.unwrapped.car_centre[1])
    color_new = env.unwrapped.track_map.get_at((x_new, y_new))
    print(f"🚗 ПОСЛЕ ШАГА 1: Центр={env.unwrapped.car_centre}, Цвет={color_new}")
    print(f"📊 Награда: {reward:.2f}, Done: {terminated}")

    if terminated:
        if color_new[0] > 250 and color_new[1] > 250 and color_new[2] > 250:
            print("💥 ВЫВОД: Машина на первом шаге уехала на БЕЛЫЙ пиксель (обочина).")
            print(
                "💡 РЕШЕНИЕ: Измените начальный угол в reset() или сдвиньте точку спавна на прямой участок дороги."
            )
        else:
            print(
                "💥 ВЫВОД: Машина разбилась по другой причине или вылетела за границы карты."
            )

env.close()
