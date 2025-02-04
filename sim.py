from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from ver3 import load_simulation_data


def animate_bodies(filename, t, y_true):
    fig, ax = plt.subplots(figsize=(8, 8))
    num_bodies = y_true.shape[0] // 4

    x_min, x_max = np.min(y_true[0::4, :]), np.max(y_true[0::4, :])
    y_min, y_max = np.min(y_true[1::4, :]), np.max(y_true[1::4, :])
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Движение тел в системе")
    ax.grid()

    lines = [ax.plot([], [], 'o-', markersize=5, lw=2, label=f"Тело массой {i + 1}")[0] for i in range(num_bodies)]

    connections = [(0, 2), (1, 2), (2, 3)]
    springs = [ax.plot([], [], '-', color='gray', lw=1.5)[0] for _ in range(len(connections))]

    ax.legend()

    # Функция инициализации
    def init():
        for line in lines:
            line.set_data([], [])
        for spring in springs:
            spring.set_data([], [])
        return lines + springs

    # Функция обновления анимации
    def update(frame):
        for i, line in enumerate(lines):
            x_idx = i * 4
            x, y = y_true[x_idx, frame], y_true[x_idx + 1, frame]
            line.set_data([x], [y])

        for i, (a, b) in enumerate(connections):
            spring_x = [y_true[a * 4, frame], y_true[b * 4, frame]]
            spring_y = [y_true[a * 4 + 1, frame], y_true[b * 4 + 1, frame]]
            springs[i].set_data(spring_x, spring_y)

        return lines + springs

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=50)

    writergif = animation.PillowWriter(fps=30)
    ani.save(f'{filename}.gif', writer=writergif)

    plt.show()


if __name__ == "__main__":
    npz_file = "simulation_data.npz"
    data = load_simulation_data(npz_file)

    if data is not None:
        t = data["t"]
        y_true = data["y_true"]
        y_obs = data["y_obs"]
        y_restored = data["y_restored"]
        K_estimated = data["K_estimated"]
        K_true = data["K_true"]

    animate_bodies("true_bodies_motion",t, y_true)
    animate_bodies("restored_bodies_motion", t, y_restored)