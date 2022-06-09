import fym
import numpy as np
from numpy import cos, sin, tan
from scipy.spatial.transform import Rotation


def frac(x, fr):
    return np.abs(x) ** fr


def sign(x, alpha=10, eps=1e-1):
    res = np.sign(x)
    crit = np.abs(x) < eps
    res[crit] = 2 * (1 / (1 + np.exp(-alpha * x[crit])) - 1 / 2)
    return res


def cross(x, y):
    return np.cross(x, y, axis=0)


def get_angles(R):
    return Rotation.from_matrix(R).as_euler("ZYX")[::-1, None]


class Line:
    def __init__(self, *args):
        self.times = [arg[0] for arg in args]
        self.points = np.hstack([arg[1] for arg in args])
        self.points_deriv = np.diff(self.points) / np.diff(self.times)

    def get(self, t):
        i = np.flatnonzero(t >= self.times)[-1]
        if i == len(self.times) - 1:
            return self.points[:, i : i + 1]
        else:
            point = self.points[:, i] + self.points_deriv[:, i] * (t - self.times[i])
            return point[:, None]

    def get_deriv(self, t):
        i = np.flatnonzero(t >= self.times)[-1]
        if i == len(self.times) - 1:
            return np.zeros_like(self.points[:, -1][:, None])
        else:
            return self.points_deriv[:, i : i + 1]


class NDE(fym.BaseEnv):
    w1 = w2 = 50
    zeta = 0.8

    def __init__(self, init=None, shape=(1, 1)):
        super().__init__()
        self.x1 = fym.BaseSystem(init, shape=shape)
        self.x2 = fym.BaseSystem(init, shape=shape)
        self.x3 = fym.BaseSystem(init, shape=shape)

    def set_dot(self, s):
        x1, x2, x3 = self.observe_list()

        self.x1.dot = x2
        self.x2.dot = x3
        self.x3.dot = (
            (s - x1) * self.w2**2 * self.w1
            - (self.w2**2 + 2 * self.zeta * self.w2 * self.w1) * x2
            - (2 * self.zeta * self.w2 + self.w1) * x3
        )


class LPF(fym.BaseSystem):
    def __init__(self, initial_state, tau=1e-2):
        super().__init__(initial_state)
        self.tau = tau

    def set_dot(self, xc):
        self.dot = -1 / self.tau * (self.state - xc)


class NTSMC(fym.BaseEnv):
    betatil = 0.1
    ptil = 5
    qtil = 3
    MO = 1
    MObar = 1

    betabar = 0.1
    pbar = 5
    qbar = 3
    MI = 1
    MIbar = 1

    def __init__(self, plant):
        super().__init__()
        self.nde_pos = NDE(shape=(3, 1))
        self.nde_angles = NDE(shape=(3, 1))
        self.nde_omega = NDE(shape=(3, 1))
        self.lpf_u = LPF(np.vstack((plant.m * plant.g / 3, 0, 0)))
        # self.nde_xId = NDE(shape=(3, 1))

        self.plant = plant

        # Reference
        self.xd = Line(
            (0, 1 * np.ones((3, 1))),
            (20, 1 * np.ones((3, 1))),
            (25, 4 * np.ones((3, 1))),
            (35, 4 * np.ones((3, 1))),
            (40, 2 * np.ones((3, 1))),
        )

    def get_u(self, t):
        # Plant states
        pos, vel, R, omega, rs = self.plant.observe_list()
        angles = get_angles(R)
        phi, theta, psi = angles.ravel()
        z = pos.ravel()[-1]
        vz = vel.ravel()[-1]
        p, q, r = omega.ravel()

        u = self.plant.rs2u(t, rs)

        # NDE states
        poshat, poshat_dot, poshat_ddot = self.nde_pos.observe_list()
        angleshat, angleshat_dot, _ = self.nde_angles.observe_list()
        omegahat, omegahat_dot, _ = self.nde_omega.observe_list()

        # Outer loop
        xO = pos[:2]
        xO_dot = vel[:2]
        xd, xd_dot, xd_ddot = self.reference(t)
        xOd, xOd_dot, xOd_ddot = xd[:2], xd_dot[:2], xd_ddot[:2]

        u1 = u[0]
        phihat, thetahat, psihat = angleshat.ravel()
        vhatxy = poshat_dot[:2]
        ahatxy = poshat_ddot[:2]

        R3 = np.array(
            [
                [sin(psi), -cos(psi)],
                [cos(psi), sin(psi)],
            ]
        )

        # Sliding surface
        SO = xO - xOd + 1 / self.betatil * frac(xO_dot - xOd_dot, self.ptil / self.qtil)

        Fhat = self.plant.m * ahatxy - (
            u1
            * np.vstack(
                (
                    cos(phihat) * cos(psihat) * sin(thetahat)
                    + sin(phihat) * sin(psihat),
                    cos(phihat) * sin(psihat) * sin(thetahat)
                    - sin(phihat) * cos(psihat),
                )
            )
            - self.plant.kt * vhatxy
        )
        util = (
            -self.plant.m
            / u1
            * R3
            @ (
                -self.plant.kt / self.plant.m * xO_dot
                + 1 / self.plant.m * Fhat
                - xOd_ddot
                + self.betatil
                * self.qtil
                / self.ptil
                * frac(xO_dot - xOd_dot, 2 - self.ptil / self.qtil)
                + self.MO * sign(SO)
                + self.MObar * np.diag(np.abs(Fhat.ravel())) @ sign(SO)
            )
        )
        zd = xd[2]
        xId = np.vstack((zd, util))
        xId_dot = xId_ddot = np.zeros_like(xId)

        # Inner loop
        xI = np.vstack((z, phi, theta))
        xI_dot = np.vstack((vz, p, q))

        xbar = np.vstack([vz, z, omega, angles[:2]])
        xbarhat = np.vstack((poshat_dot[2], poshat[2], omegahat, angleshat[:2]))
        xbarhat_dot = np.vstack(
            (poshat_ddot[2], poshat_dot[2], omegahat_dot, angleshat_dot[:2])
        )

        uhat = self.lpf_u.state

        Jxbar = np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
                [0, sin(phi) * tan(theta), cos(phi)],
                [0, cos(phi) * tan(theta), -sin(phi)],
                [
                    0,
                    q * cos(phi) * tan(theta) - r * sin(phi) * tan(theta),
                    -q * sin(phi) - r * cos(phi),
                ],
                [0, (q * sin(phi) + r * cos(phi)) / cos(theta) ** 2, 0],
            ]
        ).T

        SI = xI - xId + 1 / self.betabar * frac(xI_dot - xId_dot, self.pbar / self.qbar)

        ghatxbar = xbarhat_dot - self.f(xbarhat) - self.h(xbarhat) @ uhat

        uc = -np.linalg.inv(Jxbar @ self.h(xbar)) @ (
            self.betabar
            * self.qbar
            / self.pbar
            * frac(xI_dot - xId_dot, 2 - self.pbar / self.qbar)
            + Jxbar @ (self.f(xbar) + ghatxbar)
            - xId_ddot
            + self.MI * sign(SI)
            + self.MIbar * np.max(np.abs(Jxbar @ ghatxbar)) * sign(SI)
        )
        uc = np.insert(uc, 1, 0, axis=0)
        int_info = dict(xOd=xOd, Xd=np.vstack((xId, 0)))
        # if t > 0.3:
        #     breakpoint()
        return uc, int_info

    def f(self, xbar):
        # xbar: [vz, z, p, q, r, phi, theta, psi]
        vz, _, p, q, r, phi, theta = xbar.ravel()

        omega = np.vstack((p, q, r))

        Rr = np.array(
            [
                [1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
                [0, cos(phi), -sin(phi)],
                [0, sin(phi) / cos(theta), cos(phi) / cos(theta)],
            ]
        )
        Theta_dot = Rr @ omega

        return np.vstack(
            (
                -self.plant.g - self.plant.kt / self.plant.m * vz,
                vz,
                self.plant.Jinv
                @ (-self.plant.K2 @ omega - cross(omega, self.plant.J @ omega)),
                Theta_dot[:2],
            )
        )

    def h(self, xbar):
        vz, z, p, q, r, phi, theta = xbar.ravel()

        # Plant paramters
        Jx, Jy, Jz = np.diag(self.plant.J)

        l = self.plant.l
        d = self.plant.d

        return np.array(
            [
                [cos(phi) * cos(theta) / self.plant.m, 0, 0],
                [0, 0, 0],
                [l / (2 * Jx), 0, -l / (2 * Jx * d)],
                [0, 1 / Jy, 0],
                [0, 0, 1 / Jz],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

    def reference(self, t):
        xd = self.xd.get(t)
        xd_dot = self.xd.get_deriv(t)
        xd_ddot = np.zeros((3, 1))
        return xd, xd_dot, xd_ddot

    def set_dot(self, t, u, int_info):
        self.nde_pos.set_dot(self.plant.pos.state)
        self.nde_angles.set_dot(get_angles(self.plant.R.state))
        self.nde_omega.set_dot(self.plant.omega.state)
        self.lpf_u.set_dot(np.delete(u, 1, axis=0))
        cntr_info = int_info
        return cntr_info
