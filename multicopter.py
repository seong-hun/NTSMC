import fym
import numpy as np


def cross(x, y):
    return np.cross(x, y, axis=0)


def hat(x):
    x1, x2, x3 = x.ravel()
    return np.array(
        [
            [0, -x3, x2],
            [x3, 0, -x1],
            [-x2, x1, 0],
        ]
    )


class Multicopter(fym.BaseEnv):
    """The multicopter model.

    Configuration
    -------------

             ^ y
             |
            (0)
    x <- (3) + (1)
            (2)
             |

    """

    g = 9.8  # [m/s2]
    m = 0.5  # [kg]
    J = np.diag([5.9e-3, 5.9e-3, 1.16e-2])  # [kg.m2]
    Jr = 1.5e-5  # not found
    b = 3.13e-5  # [N.s2 / rad2]
    kt = kr = 6e-3  # [N.m.s2 / rad2]
    l = 0.255  # [m]
    K1 = np.diag([kt, kt, kt])  # [N.s/m]
    K2 = np.diag([kr, kr, kr])  # [N.m.s/rad]
    tau = 0.001  # [sec]

    Jinv = np.linalg.inv(J)
    e3 = np.vstack((0, 0, 1))

    rotorspeed_min = 300 * 2 * np.pi / 60  # [rad/s]
    rotorspeed_max = 4000 * 2 * np.pi / 60  # [rad/s]

    # Control allocation matrix
    # u = [F; M]
    d = 4.8e-3
    Ar = np.array(
        [
            [1, 1, 1, 1],
            [0, -l, 0, l],
            [-l, 0, l, 0],
            [d, -d, d, -d],
        ]
    )

    Arb = np.delete(Ar, 1, axis=0)
    Arb = np.delete(Arb, 1, axis=1)
    Arbinv = np.linalg.inv(Arb)  # F = Arinv @ u
    Br = Arbinv / b  # rotorspeed = sqrt(Br @ u)

    # Fault
    Lambda = np.diag([1, 0, 1, 1])  # LoE matrix
    # actual input: u = Ar @ Lambda @ Arinv @ uc

    def __init__(self):
        super().__init__()
        self.pos = fym.BaseSystem(np.zeros((3, 1)))
        self.vel = fym.BaseSystem(np.zeros((3, 1)))
        self.R = fym.BaseSystem(np.eye(3))
        self.omega = fym.BaseSystem(np.zeros((3, 1)))

        self.rs = fym.BaseSystem(np.zeros((4, 1)))

        self.nrotors = 4

        self.fault_index = ()

    def deriv(self, pos, vel, R, omega, u):
        F, M = u[:1], u[1:]

        m, g, J, e3 = self.m, self.g, self.J, self.e3

        dpos = vel
        dvel = (
            -m * g * e3  # gravitational force
            + F * R @ e3  # force by propellers (F: positive = upper)
            - self.K1 @ vel  # drag force
        ) / m
        dR = R @ hat(omega)
        domega = self.Jinv @ (
            -cross(omega, J @ omega)
            + M  # moment by propellers
            - self.K2 @ omega  # drag moment (K2: negative elemts.)
        )

        return dpos, dvel, dR, domega

    def set_dot(self, t, uc):
        pos, vel, R, omega, rs = self.observe_list()

        u = self.rs2u(t, rs)
        dots = self.deriv(pos, vel, R, omega, u)
        self.pos.dot, self.vel.dot, self.R.dot, self.omega.dot = dots

        rs_c = self.u2rs(uc)
        self.rs.dot = -1 / self.tau * (rs - rs_c)

        rotorforces = self.b * rs**2
        faultforces = self.get_Lambda(t) @ rotorforces

        return dict(
            uc=uc,
            u=u,
            rs=rs,
            rs_c=rs_c,
            rotorforces=rotorforces,
            faultforces=faultforces,
        )

    def get_Lambda(self, t):
        return self.Lambda

    def rs2u(self, t, rs):
        rotorspeed = np.clip(rs, self.rotorspeed_min, self.rotorspeed_max)
        rotorforces = self.b * rotorspeed**2
        faultforces = self.get_Lambda(t) @ rotorforces
        u = self.Ar @ faultforces
        return u

    def u2rs(self, u):
        ucb = np.delete(u, 1, axis=0)
        rs_cb = np.sqrt(np.clip(self.Br @ ucb, 0, None))
        rs_cb_clipped = np.clip(rs_cb, self.rotorspeed_min, self.rotorspeed_max)
        rs = np.insert(rs_cb_clipped, 1, 0, axis=0)
        return rs
