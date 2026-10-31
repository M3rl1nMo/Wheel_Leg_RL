"""平地速度指令接口。项目根目录运行 python -m RL.v4_rsl_pd.velocity_control。"""
import numpy as np

from .environment import StandingEnv, robot_info_update


class BodyVelocityController:
    """在模型复位后创建；每个物理步调用 torques(v, w)，输出顺序与 data.ctrl 相同。"""

    def __init__(self, env: StandingEnv, wheel_kd=0.001):
        self.env = env
        self.wheel_kd = wheel_kd  # N·m/(rad/s²)，轮速环对测量值做微分
        model, data = env.model, env.data
        right, left = (model.geom(f"{s}_wheel_geom") for s in ("right", "left"))
        body_y = data.xmat[env.torso].reshape(3, 3)[:, 1]
        self.track = float((data.geom_xpos[left.id] - data.geom_xpos[right.id]) @ body_y)
        self.radii = np.array([right.size[0], left.size[0]])
        joints = [model.joint(f"{s}_wheel_torque") for s in ("right", "left")]
        axes_y = np.array([data.xaxis[j.id] @ body_y for j in joints])
        if self.track <= 0 or min(self.radii) <= 0 or min(abs(axes_y)) < 0.99:
            raise ValueError("请在机器人复位后检查轮距、半径与轮轴方向")
        self.signs = np.sign(axes_y)
        self.wheel_dofs = [int(j.dofadr[0]) for j in joints]

        self.last_time = None
        self.last_omega = np.zeros(2)
        self.alpha = np.zeros(2)

    def motor_speed_targets(self, v, w):
        """v 为车体前向速度 m/s，w 为左转角速度 rad/s；返回编码器坐标的 rad/s。"""
        if not np.isfinite([v, w]).all():
            raise ValueError("速度指令必须有限")
        rim_speed = np.array([v + w * self.track / 2, v - w * self.track / 2])
        wheels = self.signs * rim_speed / self.radii
        return np.concatenate([np.zeros(4), wheels])

    def torques(self, v, w):
        """腿保持名义角度；轮子使用速度 P 和实测角加速度阻尼，最后限幅。"""
        targets = self.motor_speed_targets(v, w)
        return self.target_torques(self.env.q_nominal, targets[4:])

    def target_torques(self, q_target, wheel_target):
        env = self.env
        leg = env.leg_torques(q_target, robot_info_update(env.data))
        omega = env.data.qvel[self.wheel_dofs]
        now = env.data.time
        if self.last_time is None or now < self.last_time or now == 0:
            self.alpha.fill(0)
        elif now > self.last_time:
            self.alpha = (omega - self.last_omega) / (now - self.last_time)
        self.last_time = now
        self.last_omega = omega.copy()
        wheel = env.cfg["kv"] * (wheel_target - omega) - self.wheel_kd * self.alpha
        return np.concatenate([leg, np.clip(wheel, -1.5, 1.5)])


if __name__ == "__main__":
    env = StandingEnv()
    control = BodyVelocityController(env)
    for v, w in [(0.15, 0), (-0.15, 0), (0, 0.5), (0, -0.5), (0, 0)]:
        print(f"v={v:+.2f}, w={w:+.2f}:",
              "目标 rad/s", control.motor_speed_targets(v, w).round(3),
              "力矩 N·m", control.torques(v, w).round(3))
