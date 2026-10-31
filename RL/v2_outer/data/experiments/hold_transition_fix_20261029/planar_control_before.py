"""平面目标外环及 LQR 教师；坐标约定 x向前、y向左、yaw绕z。"""

# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
import math
import numpy as np
from RL.common.lqr_imitation import LQRTeacher
from robot_control_main import control_left_leg, control_right_leg


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PlanarLQRTeacher(LQRTeacher):
    def action(self, info, yaw_rate, velocity, turn_rate):
        lengths = self.bal.compute_leg_lengths(current_roll=-info['euler']['roll'], target_roll=0., base_length=.1)
        speed = (info['vel']['leftwheel']-info['vel']['rightwheel'])/2
        pos = (info['pos']['leftwheel']-info['pos']['rightwheel'])/2
        self.ll.x_b_set = self.lr.x_b_set = pos*.06
        self.ll.v_set = self.lr.v_set = velocity
        args = (-info['euler']['pitch'], -info['gyro_pitch'], yaw_rate, turn_rate)
        lc = control_left_leg(info,self.left,self.ll,self.yl,self.pl,*args,lengths[0],pos,speed,.06,False,False)
        rc = control_right_leg(info,self.right,self.lr,self.yr,self.pr,*args,lengths[1],pos,speed,.06,False,False)
        return np.array([rc[0],rc[1],lc[0],lc[1],rc[2],lc[2]],dtype=np.float32)


class PlanarPositionController:
    def __init__(self, initial_yaw=0.):
        self.integral = 0.
        self.heading = initial_yaw
        self.holding = False

    def update(self, xy, yaw, target, dt):
        dx,dy = np.asarray(target)-np.asarray(xy)
        forward=math.cos(yaw)*dx+math.sin(yaw)*dy
        left=-math.sin(yaw)*dx+math.cos(yaw)*dy
        distance=math.hypot(dx,dy)
        # 近目标冻结目标朝向，避免距离很小时目标方向剧烈改变。
        if self.holding and distance>.15:
            self.holding=False
        elif not self.holding and distance<.025:
            self.holding=True
            self.heading=yaw
        if not self.holding:
            self.heading=math.atan2(dy,dx)
        angle=wrap_angle(self.heading-yaw)
        alignment=max(0.,math.cos(angle))
        if abs(angle)<.6 or self.holding:
            self.integral=float(np.clip(self.integral+forward*dt,-.2,.2))
        velocity=float(np.clip(forward+.6*self.integral,-.15,.15))
        if not self.holding:
            velocity=max(0.,velocity)*alignment
        turn=float(np.clip(1.5*angle,-.6,.6))
        return forward,left,velocity,turn
