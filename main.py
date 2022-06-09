import fym
import matplotlib.pyplot as plt
import numpy as np
from ntsmc import NTSMC
from scipy.spatial.transform import Rotation

from multicopter import Multicopter


class Env(fym.BaseEnv):
    def __init__(self):
        super().__init__(max_t=20, dt=0.001)
        self.plant = Multicopter()
        self.cntr = NTSMC(self.plant)

    def step(self):
        R = self.plant.R.state
        angles = np.rad2deg(Rotation.from_matrix(R).as_euler("ZYX")[::-1])
        done = self.update()[-1]  # return done
        return done or np.any(np.abs(angles) > 60)

    def set_dot(self, t):
        u, int_info = self.cntr.get_u(t)

        plant_info = self.plant.set_dot(t, u)
        cntr_info = self.cntr.set_dot(t, u, int_info)

        return dict(
            t=t,
            **self.observe_dict(),
            u=u,
            plant_info=plant_info,
            cntr_info=cntr_info,
        )


def run():
    env = Env()
    env.logger = fym.Logger("data.h5")

    env.reset()
    while True:
        env.render()
        if env.step():
            break
    env.close()


def plot():
    # Data processing
    data = fym.load("data.h5")
    pos = data["plant"]["pos"]
    ref_pos = data["cntr_info"]["xOd"]

    angles = Rotation.from_matrix(data["plant"]["R"]).as_euler("ZYX")
    angles = np.rad2deg(angles)
    ref_angles = np.rad2deg(data["cntr_info"]["Xd"][:, 1:4])
    ref_z = data["cntr_info"]["Xd"][:, 0]

    # Plotting
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"

    style = dict(c="k", ls="--")
    refstyle = dict(c="r")

    fig = plt.figure()
    axes = fig.subplots(2, 1, sharex=True)

    ax = axes[0]
    ax.plot(data["t"], ref_pos[:, 0], label="Reference", **refstyle)
    ax.plot(data["t"], pos[:, 0], label="NTSMC", **style)
    ax.legend()
    ax.set_ylabel(r"$x$, m")

    ax = axes[1]
    ax.plot(data["t"], ref_pos[:, 1], **refstyle)
    ax.plot(data["t"], pos[:, 1], **style)
    ax.set_ylabel(r"$y$, m")

    fig = plt.figure()
    axes = fig.subplots(2, 2, sharex=True)

    ax = axes[0, 0]
    ax.plot(data["t"], ref_angles[:, 0], label="Reference", **refstyle)
    ax.plot(data["t"], angles[:, 2], label="NTSMC", **style)
    ax.legend()
    ax.set_ylabel(r"$\phi$, deg")

    ax = axes[0, 1]
    ax.plot(data["t"], ref_angles[:, 1], **refstyle)
    ax.plot(data["t"], angles[:, 1], **style)
    ax.set_ylabel(r"$\theta$, deg")

    ax = axes[1, 0]
    ax.plot(data["t"], ref_angles[:, 2], **refstyle)
    ax.plot(data["t"], angles[:, 0], **style)
    ax.set_ylabel(r"$\psi$, deg")
    ax.set_xlabel("Time, sec")

    ax = axes[1, 1]
    ax.plot(data["t"], ref_z, **refstyle)
    ax.plot(data["t"], pos[:, 2], **style)
    ax.set_ylabel("Altitude, m")
    ax.set_xlabel("Time, sec")

    fig.suptitle("Tracking results")
    fig.tight_layout()

    fig = plt.figure()
    axes = fig.subplots(2, 2, sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(data["t"], data["plant_info"]["faultforces"][:, i], **style)
        ax.set_ylabel(rf"$f_{i + 1}$, N")
        if i in (2, 3):
            ax.set_xlabel("Time, sec")

    fig.suptitle("Rotor forces")
    fig.tight_layout()

    plt.show()


def main():
    run()
    plot()


if __name__ == "__main__":
    main()
